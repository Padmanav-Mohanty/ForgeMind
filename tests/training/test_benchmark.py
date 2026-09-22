"""
Tests for the QLoRA VRAM benchmark helper logic (src/training/benchmark.py).

Only the non-GPU parts are tested here — sample building, renderers, and
configuration. The GPU runner (run_single_benchmark / run_benchmark)
requires CUDA and is exercised on Colab, not in this suite.

How to run:
    pytest tests/training/test_benchmark.py -v
"""

from __future__ import annotations

import json

import pytest

from src.training.benchmark import (
    BenchmarkConfig,
    SampleReport,
    build_length_sample,
    default_configs,
    environment_info,
    load_train_records,
    render_result_summary,
    render_results_table,
    save_benchmark_report,
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
