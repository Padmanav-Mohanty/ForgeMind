"""
Tests for the QLoRA VRAM benchmark helper logic (src/training/benchmark.py).

Only the non-GPU parts are tested here — sample building, renderers, and
configuration. The GPU runner (run_single_benchmark / run_benchmark)
requires CUDA and is exercised on Colab, not in this suite.

How to run:
    pytest tests/training/test_benchmark.py -v
"""

from __future__ import annotations

import inspect
import json

import pytest

from src.training.benchmark import (
    SWEEP_SEQ_LENGTHS,
    BenchmarkConfig,
    StepResult,
    build_length_sample,
    compute_record_lengths,
    configure_padding,
    default_configs,
    environment_info,
    load_train_records,
    phases_completed,
    render_result_summary,
    render_results_table,
    render_sweep_table,
    run_single_benchmark,
    save_benchmark_report,
    sweep_configs,
)
from src.data.utils import PROJECT_ROOT, build_example


def _make_record(user: str, assistant: str, index: int) -> dict:
    return build_example(
        user,
        assistant,
        {"source": "test", "task": "debugging", "group_key": f"g{index}"},
    )


# --- configuration ---------------------------------------------------------


class TestBenchmarkConfig:
    def test_default_matrix_is_2048_and_4096_at_batch_1(self):
        configs = default_configs()
        assert [c.max_seq_length for c in configs] == [2048, 4096]
        assert all(c.batch_size == 1 for c in configs)
        assert all(c.quantization == "4-bit" for c in configs)
        assert all(c.gradient_checkpointing is True for c in configs)
        assert all(c.packing is False for c in configs)

    def test_max_seq_length_not_preset_in_repo_config(self):
        # the repo's qlora.yaml must stay unset until the benchmark results
        # are in — the benchmark never writes to it
        config_text = (PROJECT_ROOT / "configs" / "qlora.yaml").read_text(encoding="utf-8")
        assert "max_seq_length: null" in config_text


# --- deterministic sample building -----------------------------------------


class TestBuildLengthSample:
    @pytest.fixture()
    def records(self) -> list[dict]:
        # lengths roughly: 300, 500, 800, 1000, 1300, 1600, 1900, 2200 chars
        # (stub tokenizer counts words + framing; keep them distinguishable)
        records = []
        for words in (40, 70, 110, 150, 200, 260, 320, 400):
            records.append(_make_record("word " * words, "answer " * 20, len(records)))
        return records

    def _stub_tokenizer(self):
        from tests.tokenization.test_render_and_analysis import StubTokenizer

        return StubTokenizer()

    def test_deterministic_across_calls(self, records):
        tokenizer = self._stub_tokenizer()
        sample_a, report_a = build_length_sample(records, tokenizer, 300, count=4, seed=42)
        sample_b, report_b = build_length_sample(records, tokenizer, 300, count=4, seed=42)
        assert sample_a == sample_b
        assert report_a.selection == report_b.selection

    def test_prefers_near_cap_band(self, records):
        tokenizer = self._stub_tokenizer()
        sample, report = build_length_sample(records, tokenizer, 350, count=4, seed=42)
        assert report.selection["near_cap_0.8-1.0"] >= 1
        assert report.near_cap_count >= 1
        # every selected example must fit under the cap
        assert report.token_lengths["sample_max"] <= 350

    def test_skips_over_cap_examples(self, records):
        tokenizer = self._stub_tokenizer()
        # cap below everything: nothing fits, fallback engages
        sample, report = build_length_sample(records, tokenizer, 10, count=2, seed=42)
        assert report.skipped_over_cap == len(records)
        assert report.selected <= 2

    def test_never_selects_over_cap(self, records):
        tokenizer = self._stub_tokenizer()
        sample, _ = build_length_sample(records, tokenizer, 350, count=4, seed=42)
        assert all(
            item["messages"][0]["content"] for item in sample
        )
        assert len(sample) <= 4

    def test_sample_report_serializes(self, records):
        tokenizer = self._stub_tokenizer()
        _, report = build_length_sample(records, tokenizer, 350, count=4, seed=42)
        payload = json.dumps(report.to_dict())  # must be JSON-serializable
        assert json.loads(payload)["max_seq_length"] == 350


# --- renderers ---------------------------------------------------------------


class TestRenderers:
    def _result(self, **overrides) -> dict:
        config = BenchmarkConfig(max_seq_length=2048)
        result = {
            "config": vars(config) | {"label": config.label()},
            "environment": {"gpu_name": "Tesla T4", "gpu_vram_gb": 15.0},
            "steps": {
                "forward": "PASS",
                "backward": "PASS",
                "training_step": "PASS",
                "cuda_oom": False,
                "peak_vram_gb": 9.4,
                "step_time_seconds": 1.23,
                "notes": [],
            },
        }
        result["steps"].update(overrides.pop("steps", {}))
        return result

    def test_summary_matches_requested_format(self):
        text = render_result_summary(self._result())
        assert "Sequence length: 2048" in text
        assert "Batch size: 1" in text
        assert "Quantization: 4-bit" in text
        assert "Gradient checkpointing: enabled" in text
        assert "Peak VRAM: 9.4 GB" in text
        assert "CUDA OOM: NO" in text
        assert "Forward pass: PASS" in text
        assert "Backward pass: PASS" in text
        assert "Training step: PASS" in text
        assert "Approximate step time: 1.23 seconds" in text

    def test_summary_reports_oom(self):
        result = self._result(
            steps={
                "forward": "FAIL (CUDA OOM)",
                "backward": "SKIP",
                "training_step": "SKIP",
                "cuda_oom": True,
                "peak_vram_gb": 15.1,
                "step_time_seconds": None,
                "notes": ["forward pass OOM"],
            }
        )
        text = render_result_summary(result)
        assert "CUDA OOM: YES" in text
        assert "Forward pass: FAIL (CUDA OOM)" in text
        assert "Note: forward pass OOM" in text

    def test_table_row_per_configuration(self):
        results = [
            self._result(),
            self._result(steps={"peak_vram_gb": None, "step_time_seconds": None,
                                "training_step": "FAIL (CUDA OOM)", "cuda_oom": True}),
        ]
        table = render_results_table(results)
        lines = table.strip().splitlines()
        assert len(lines) == 4  # header, separator, two rows
        assert "| 2048 | 1 | no | 9.4 | NO | PASS | PASS | PASS | 1.23 |" in table
        assert "n/a" in table and "YES" in table

    def test_save_benchmark_report_writes_json(self, tmp_path):
        report = {"results": [], "model_id": "m"}
        path = save_benchmark_report(report, path=tmp_path / "r.json")
        assert json.loads(path.read_text(encoding="utf-8"))["model_id"] == "m"


# --- tokenizer padding configuration (offline, stub tokenizer) --------------


class _LlamaLikeTokenizer:
    """Mimics Llama 3.2's tokenizer: no pad token, an EOS token.

    Mutable attributes like the real PreTrainedTokenizer, but no model
    download and no network.
    """

    def __init__(self):
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<|end_of_text|>"
        self.eos_token_id = 128001
        self.padding_side = "right"  # the HF default the benchmark must override


class TestConfigurePadding:
    def test_pad_token_set_to_eos(self):
        tok = _LlamaLikeTokenizer()
        info = configure_padding(tok)
        assert tok.pad_token == tok.eos_token == "<|end_of_text|>"
        assert tok.pad_token_id == tok.eos_token_id
        assert info["pad_token_is_eos"] is True
        assert info["pad_token_id"] == 128001

    def test_padding_side_is_left(self):
        tok = _LlamaLikeTokenizer()
        info = configure_padding(tok)
        assert tok.padding_side == "left"
        assert info["padding_side"] == "left"

    def test_explicit_pad_token_is_respected_not_overwritten(self):
        tok = _LlamaLikeTokenizer()
        tok.pad_token = "<|finetune_right_pad_id|>"
        tok.pad_token_id = 128004
        info = configure_padding(tok)
        assert tok.pad_token == "<|finetune_right_pad_id|>"
        assert info["pad_token_is_eos"] is False
        assert tok.padding_side == "left"  # side is still forced

    def test_neither_pad_nor_eos_raises_actionable_error(self):
        tok = _LlamaLikeTokenizer()
        tok.eos_token = None
        tok.eos_token_id = None
        with pytest.raises(ValueError, match="pad_token"):
            configure_padding(tok)

    def test_idempotent(self):
        tok = _LlamaLikeTokenizer()
        first = configure_padding(tok)
        second = configure_padding(tok)
        assert first == second
        assert tok.pad_token == tok.eos_token

    def test_benchmark_imports_without_cuda(self):
        # regression guard for the lazy-import contract: the padding helpers
        # must be reachable with no GPU and no torch import at module load
        import src.training.benchmark as bench

        assert callable(bench.configure_padding)
        assert callable(bench.phases_completed)


# --- VRAM reporting gate (pure logic, no GPU) --------------------------------


class TestVramGate:
    def test_gate_requires_all_three_phases(self):
        assert phases_completed(
            StepResult(forward="PASS", backward="PASS", training_step="PASS")
        )
        # any phase short of PASS invalidates the measurement
        assert not phases_completed(
            StepResult(forward="FAIL (CUDA OOM)", backward="SKIP", training_step="SKIP")
        )
        assert not phases_completed(
            StepResult(forward="PASS", backward="FAIL (CUDA OOM)", training_step="SKIP")
        )
        assert not phases_completed(StepResult())  # all SKIP: nothing ran

    def test_failed_run_withholds_peak_vram_in_summary(self):
        # the Colab failure mode: benchmark died before forward, yet a
        # weights-load peak was reported as if it were a training measurement
        config = BenchmarkConfig(max_seq_length=2048)
        result = {
            "config": vars(config) | {"label": config.label()},
            "environment": {},
            "steps": {
                "forward": "SKIP",
                "backward": "SKIP",
                "training_step": "SKIP",
                "cuda_oom": False,
                "peak_vram_gb": None,
                "step_time_seconds": None,
                "notes": [
                    "error: ValueError: Asking to pad but the tokenizer does "
                    "not have a padding token",
                    "training phases did not complete — Peak VRAM withheld "
                    "(weights-load peak was 3.56 GB, not a training measurement)",
                ],
            },
        }
        text = render_result_summary(result)
        assert "Peak VRAM: n/a (required phases did not complete)" in text
        assert "Peak VRAM: 3.56 GB" not in text  # the load peak must not leak through
        assert "Note: training phases did not complete" in text

    def test_successful_run_still_reports_peak_vram(self):
        result = {
            "config": vars(BenchmarkConfig(max_seq_length=2048))
            | {"label": "seq=2048 bs=1 pack=False"},
            "environment": {},
            "steps": {
                "forward": "PASS",
                "backward": "PASS",
                "training_step": "PASS",
                "cuda_oom": False,
                "peak_vram_gb": 9.41,
                "step_time_seconds": 1.2,
                "notes": [],
            },
        }
        assert "Peak VRAM: 9.41 GB" in render_result_summary(result)


# --- forward-pass sequence-length sweep (offline logic) ----------------------


class TestSweep:
    @pytest.fixture()
    def records(self) -> list[dict]:
        from src.data.utils import build_example

        return [
            build_example(
                "word " * words,
                "answer " * 20,
                {"source": "test", "task": "debugging", "group_key": f"g{i}"},
            )
            for i, words in enumerate((40, 70, 110, 150, 200, 260, 320, 400))
        ]

    def test_sweep_lengths_are_512_to_1536(self):
        assert SWEEP_SEQ_LENGTHS == (512, 768, 1024, 1280, 1536)

    def test_sweep_configs_hold_everything_but_length_identical(self):
        reference = BenchmarkConfig(max_seq_length=2048)
        for config in sweep_configs():
            assert config.batch_size == reference.batch_size == 1
            assert config.packing is False
            assert config.gradient_checkpointing is True
            assert config.quantization == "4-bit"
            assert (config.lora_r, config.lora_alpha) == (16, 32)

    def test_sweep_configs_custom_lengths_preserve_order(self):
        assert [c.max_seq_length for c in sweep_configs((256, 384))] == [256, 384]

    def test_runner_defaults_to_full_phase_sequence(self):
        signature = inspect.signature(run_single_benchmark)
        assert signature.parameters["phases"].default == ("forward", "backward", "step")

    def test_phases_completed_forward_only(self):
        assert phases_completed(StepResult(forward="PASS"))  # sweep success shape
        assert not phases_completed(StepResult(forward="FAIL (CUDA OOM)"))
        # a FAIL anywhere invalidates, even with later phases SKIP
        assert not phases_completed(
            StepResult(forward="PASS", backward="FAIL (CUDA OOM)")
        )

    def test_phases_completed_requires_forward_unlike_skip_all(self):
        assert not phases_completed(StepResult())  # nothing ran

    def test_compute_record_lengths_matches_inline_computation(self, records):
        from tests.tokenization.test_render_and_analysis import StubTokenizer

        precomputed = compute_record_lengths(records, StubTokenizer())
        sample_cached, report_cached = build_length_sample(
            records, StubTokenizer(), 350, count=4, seed=42, precomputed=precomputed
        )
        sample_fresh, report_fresh = build_length_sample(
            records, StubTokenizer(), 350, count=4, seed=42
        )
        assert sample_cached == sample_fresh
        assert report_cached.to_dict() == report_fresh.to_dict()

    def test_sweep_table_one_row_per_length(self):
        def _row(length: int, *, forward: str, oom: bool, peak: float | None) -> dict:
            config = BenchmarkConfig(max_seq_length=length)
            return {
                "config": vars(config) | {"label": config.label()},
                "steps": {
                    "forward": forward,
                    "backward": "SKIP",
                    "training_step": "SKIP",
                    "cuda_oom": oom,
                    "peak_vram_gb": peak,
                    "step_time_seconds": None,
                    "notes": [],
                },
            }

        results = [
            _row(512, forward="PASS", oom=False, peak=4.51),
            _row(768, forward="PASS", oom=False, peak=4.98),
            _row(1024, forward="FAIL (CUDA OOM)", oom=True, peak=None),
        ]
        table = render_sweep_table(results)
        lines = table.strip().splitlines()
        assert len(lines) == 5  # header, separator, three rows
        assert "| 512 | 1 | PASS | NO | 4.51 |" in table
        assert "| 1024 | 1 | FAIL (CUDA OOM) | YES | n/a |" in table

    def test_sweep_table_withholds_vram_on_failed_forward(self):
        config = BenchmarkConfig(max_seq_length=1536)
        result = {
            "config": vars(config) | {"label": config.label()},
            "steps": {
                "forward": "SKIP",
                "backward": "SKIP",
                "training_step": "SKIP",
                "cuda_oom": False,
                "peak_vram_gb": None,
                "step_time_seconds": None,
                "notes": ["error: something before forward"],
            },
        }
        text = render_sweep_table([result])
        assert "| 1536 | 1 | SKIP | NO | n/a |" in text


# --- environment helpers (no GPU required, must not raise) ------------------


class TestEnvironment:
    def test_environment_info_reports_without_gpu(self):
        info = environment_info()
        assert "python" in info
        # torch may or may not be installed locally; either way no crash
        assert info.get("cuda_available") in (True, False, None) or "cuda_error" in info

    def test_load_train_records_missing_file_is_empty(self, tmp_path):
        assert load_train_records(tmp_path / "nope.jsonl") == []

    def test_load_train_records_reads_real_split(self):
        records = load_train_records()
        if not records:
            pytest.skip("data/final/train.jsonl not generated on this machine")
        assert len(records) > 0
        assert "messages" in records[0]
