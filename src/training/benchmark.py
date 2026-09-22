"""Colab-ready QLoRA VRAM feasibility benchmark (Llama-3.2-3B-Instruct).

Preparation for the QLoRA stage: measures, on the actual GPU, whether the
intended 4-bit QLoRA setup can train at candidate `max_seq_length` values
(2048 and 4096 by default — the same candidates as
`configs/data_config.yaml`).

Read-only with respect to the dataset: it reads `data/final/train.jsonl` to
build a tiny representative sample and never writes, filters, or trims any
data file. It never runs full training — a handful of optimization steps per
configuration, enough to exercise GPU memory, nothing more.

GPU imports (torch, transformers, peft, trl, bitsandbytes) are deliberately
lazy: the pure helpers (sample building, renderers, config) import cleanly
on a machine without CUDA so they can be unit-tested offline. Only the
benchmark runner needs a GPU.

Entry points:
    run_benchmark()       — full 2048 + 4096 benchmark on a GPU machine
    build_length_sample() — deterministic representative sample (pure)
    render_results_table()— markdown comparison table (pure)
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.utils import PROJECT_ROOT, load_config
from src.tokenization.render import count_tokens, extract_conversation, percentile

TRAIN_PATH = PROJECT_ROOT / "data" / "final" / "train.jsonl"
DEFAULT_TOKENIZER_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

BENCHMARK_SEED = 42
WARMUP_STEPS = 1
TIMED_STEPS = 3


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """One benchmark configuration (mirrors the intended QLoRA setup)."""

    max_seq_length: int
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    quantization: str = "4-bit"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_steps: int = WARMUP_STEPS
    timed_steps: int = TIMED_STEPS
    packing: bool = False
    seed: int = BENCHMARK_SEED

    def label(self) -> str:
        return f"seq={self.max_seq_length} bs={self.batch_size} pack={self.packing}"


def default_configs() -> list[BenchmarkConfig]:
    """The primary benchmark matrix: 2048 and 4096 at batch size 1."""
    return [BenchmarkConfig(max_seq_length=2048), BenchmarkConfig(max_seq_length=4096)]


def optional_batch2_configs() -> list[BenchmarkConfig]:
    """Optional follow-ups: only worth running after batch 1 passes."""
    return [BenchmarkConfig(max_seq_length=2048, batch_size=2)]


# ---------------------------------------------------------------------------
# Deterministic representative sample (pure — unit-testable, no tokenizer)
# ---------------------------------------------------------------------------


@dataclass
class SampleReport:
    """What build_length_sample selected and why (for the benchmark log)."""

    max_seq_length: int
    requested: int
    selected: int
    selection: dict[str, int]
    token_lengths: dict[str, int | float]
    near_cap_count: int
    skipped_over_cap: int
    available: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_seq_length": self.max_seq_length,
            "requested": self.requested,
            "selected": self.selected,
            "selection": self.selection,
            "token_lengths": self.token_lengths,
            "near_cap_count": self.near_cap_count,
            "skipped_over_cap": self.skipped_over_cap,
            "available": self.available,
        }


def build_length_sample(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    count: int = 4,
    seed: int = BENCHMARK_SEED,
) -> tuple[list[dict[str, Any]], SampleReport]:
    """Deterministically pick `count` chat-template examples near the cap.

    Purpose: exercise the memory profile of the LONGEST examples a training
    run would actually see at this sequence length — not average examples.

    Selection (seeded, no dataset mutation):
      1. token length in [0.8 * cap, cap] — representative near-cap mass
      2. remainder in [0.6 * cap, 0.8 * cap) — below-cap context
      3. fallback: any examples under the cap, longest first
      4. last resort: the full training set average (only if nothing fits)

    Examples over the cap are skipped (they will be trimmed or filtered
    before real training — that decision is separate from this benchmark).
    """
    lengths: list[tuple[int, int, list[dict[str, str]]]] = []
    for index, record in enumerate(records):
        conversation = extract_conversation(record)
        if conversation is None:
            continue
        lengths.append((index, count_tokens(conversation, tokenizer), conversation))

    under_cap = [(i, n, c) for i, n, c in lengths if n <= max_seq_length]
    over_cap = len(lengths) - len(under_cap)

    band_a = [(i, n, c) for i, n, c in under_cap if n >= 0.8 * max_seq_length]
    band_b = [(i, n, c) for i, n, c in under_cap if 0.6 * max_seq_length <= n < 0.8 * max_seq_length]

    rng = random.Random(seed)
    selection: dict[str, int] = {}
    chosen: list[tuple[int, int, list[dict[str, str]]]] = []

    take_a = min(2, len(band_a))
    chosen.extend(rng.sample(band_a, take_a) if take_a else [])
    selection["near_cap_0.8-1.0"] = take_a
    remaining = count - len(chosen)
    take_b = min(remaining, len(band_b))
    chosen.extend(rng.sample(band_b, take_b) if take_b else [])
    selection["mid_0.6-0.8"] = take_b

    if len(chosen) < count and under_cap:
        remaining = count - len(chosen)
        taken_indexes = {index for index, _, _ in chosen}
        pool = [(i, n, c) for i, n, c in under_cap if i not in taken_indexes]
        extra = sorted(pool, key=lambda item: -item[1])[:remaining]
        chosen.extend(extra)
        selection["longest_fallback"] = len(extra)

    if not chosen and lengths:
        avg = sorted(lengths, key=lambda x: abs(x[1] - (sum(n for _, n, _ in lengths) / len(lengths))))
        chosen = avg[:count]
        selection["dataset_average_fallback"] = len(chosen)

    # order deterministically: longest first, then by original index
    chosen.sort(key=lambda item: (-item[1], item[0]))
    sample = [{"messages": conversation, "train_index": index} for index, _, conversation in chosen]

    all_lengths = sorted(n for _, n, _ in lengths)
    sample_lengths = sorted(item[1] for item in chosen) or [0]
    report = SampleReport(
        max_seq_length=max_seq_length,
        requested=count,
        selected=len(sample),
        selection=selection,
        token_lengths={
            "sample_min": sample_lengths[0],
            "sample_max": sample_lengths[-1],
            "sample_mean": round(sum(sample_lengths) / max(len(sample_lengths), 1), 1),
            "dataset_p50": percentile(all_lengths, 50),
            "dataset_p90": percentile(all_lengths, 90),
            "dataset_p95": percentile(all_lengths, 95),
            "dataset_p99": percentile(all_lengths, 99),
            "dataset_max": all_lengths[-1] if all_lengths else 0,
        },
        near_cap_count=sum(1 for _, n, _ in under_cap if n >= 0.8 * max_seq_length),
        skipped_over_cap=over_cap,
        available=len(lengths),
    )
    return sample, report


def load_train_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the train split read-only; returns [] if the file is missing."""
    path = path or TRAIN_PATH
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Environment reporting / graceful degradation
# ---------------------------------------------------------------------------


def environment_info() -> dict[str, Any]:
    """Collect versions + GPU facts; missing pieces become None, not errors."""
    import platform
    import subprocess
    import sys

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate"):
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - bitsandbytes raises on missing CUDA
            info[module_name] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        info["cuda_error"] = str(exc)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        info["nvidia_smi"] = smi.stdout.strip() if smi.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = None
    return info


def preflight_errors() -> list[str]:
    """Hard blockers with actionable messages; empty list = ready to run."""
    errors: list[str] = []
    try:
        import torch

        if not torch.cuda.is_available():
            errors.append(
                "CUDA is not available. In Colab: Runtime > Change runtime type > "
                "select the T4 GPU hardware accelerator, then re-run all cells."
            )
    except ImportError:
        errors.append("PyTorch is not installed — run the installation cell first.")
        return errors
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"bitsandbytes failed to import ({exc}); 4-bit quantization cannot run."
        )
    if not TRAIN_PATH.is_file():
        errors.append(
            f"{TRAIN_PATH} not found — upload the repository (or data/final/train.jsonl) "
            "or set FORGEMIND_ROOT to the repo location before benchmarking."
        )
    return errors


# ---------------------------------------------------------------------------
# GPU benchmark runner (needs CUDA; imported lazily by the notebook)
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    forward: str = "SKIP"
    backward: str = "SKIP"
    training_step: str = "SKIP"
    oom: bool = False
    peak_vram_gb: float | None = None
    step_time_seconds: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward": self.forward,
            "backward": self.backward,
            "training_step": self.training_step,
            "cuda_oom": self.oom,
            "peak_vram_gb": self.peak_vram_gb,
            "step_time_seconds": self.step_time_seconds,
            "notes": self.notes,
        }


def _torch_imports():
    import torch

    return torch


def run_single_benchmark(
    config: BenchmarkConfig,
    tokenizer: Any,
    sample: list[dict[str, Any]],
    model_id: str = DEFAULT_MODEL_ID,
    hf_token: str | None = None,
    verbose: bool = True,
    texts_override: list[str] | None = None,
) -> dict[str, Any]:
    """Run one configuration end to end; returns its result dict.

    Catches torch.cuda.OutOfMemoryError per phase (forward, backward,
    optimizer step) so one OOM reports cleanly instead of killing the run.
    Always releases the model and CUDA cache before returning.

    texts_override replaces the per-example chat-template rendering (used by
    the optional packing cell to feed one pre-packed long sequence);
    everything else — tokenization settings, labels, memory measurement —
    is identical.
    """
    torch = _torch_imports()
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    device_map = {"": 0}
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    result: dict[str, Any] = {
        "config": vars(config) | {"label": config.label()},
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_vram_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            ),
            "compute_dtype": str(compute_dtype),
        },
        "steps": StepResult().to_dict(),
    }
    steps = StepResult()
    model = None

    def log(message: str) -> None:
        if verbose:
            print(message)

    try:
        torch.cuda.reset_peak_memory_stats(0)
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map=device_map,
            token=hf_token,
            torch_dtype=compute_dtype,
        )
        model.config.use_cache = False  # required with gradient checkpointing
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        result["load_seconds"] = round(time.perf_counter() - load_start, 1)

        # ---- batch: left-pad, truncate to the cap, mask the pad ------------
        if texts_override is not None:
            texts = texts_override[: config.batch_size]
        else:
            texts = []
            for item in sample[: config.batch_size]:
                conversation = item["messages"]
                texts.append(
                    tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=False
                    )
                )
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        tokenizer.padding_side = "left"  # pad the front: loss positions stay at the end
        encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
        )
        input_ids = encodings["input_ids"].to("cuda")
        attention_mask = encodings["attention_mask"].to("cuda")
        labels = input_ids.clone()
        labels[labels == pad_id] = -100  # never train on padding
        log(
            f"[{config.label()}] batch: {input_ids.shape[0]} x {input_ids.shape[1]} tokens "
            f"(pad token {pad_id})"
        )

        # ---- forward ------------------------------------------------------
        try:
            forward_start = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_start
            steps.forward = "PASS"
            log(f"[{config.label()}] forward PASS in {forward_seconds:.2f}s")
        except torch.cuda.OutOfMemoryError:
            steps.oom = True
            steps.forward = "FAIL (CUDA OOM)"
            steps.notes.append("forward pass OOM")

        # ---- backward on the forward graph we already hold -----------------
        if steps.forward == "PASS":
            try:
                backward_start = time.perf_counter()
                outputs.loss.backward()
                torch.cuda.synchronize()
                backward_seconds = time.perf_counter() - backward_start
                model.zero_grad(set_to_none=True)
                steps.backward = "PASS"
                log(f"[{config.label()}] backward PASS in {backward_seconds:.2f}s")
            except torch.cuda.OutOfMemoryError:
                steps.oom = True
                steps.backward = "FAIL (CUDA OOM)"
                steps.notes.append("backward pass OOM")
            finally:
                outputs = None  # the autograd graph must not outlive the phase

        # ---- optimizer step (the real training-step memory profile) -------
        if steps.backward == "PASS":
            try:
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=1e-4
                )
                model.train()
                timed = []
                for step_index in range(config.timed_steps):
                    step_start = time.perf_counter()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    outputs.loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    timed.append(time.perf_counter() - step_start)
                steps.training_step = "PASS"
                steps.step_time_seconds = round(sum(timed) / len(timed), 2)
                log(
                    f"[{config.label()}] training step PASS; "
                    f"avg {steps.step_time_seconds}s over {config.timed_steps} steps "
                    f"(warmup {config.warmup_steps} included in the {config.warmup_steps + config.timed_steps} total)"
                )
            except torch.cuda.OutOfMemoryError:
                steps.oom = True
                steps.training_step = "FAIL (CUDA OOM)"
                steps.notes.append("optimizer step OOM")

    except torch.cuda.OutOfMemoryError:
        steps.oom = True
        steps.notes.append("OOM outside a guarded phase")
    except Exception as exc:  # noqa: BLE001 - report the failure cleanly
        steps.notes.append(f"error: {type(exc).__name__}: {exc}")

    try:
        peak = torch.cuda.max_memory_allocated(0) / 1024**3
        steps.peak_vram_gb = round(peak, 2)
    except Exception:  # noqa: BLE001
        steps.peak_vram_gb = None

    result["steps"] = steps.to_dict()
    result["ok"] = (not steps.oom) and steps.training_step == "PASS"

    # ---- cleanup: always, so the next configuration starts clean ---------
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    return result


def run_benchmark(
    configs: list[BenchmarkConfig] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    hf_token: str | None = None,
    sample_count: int = 4,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full benchmark matrix and return the complete result dict."""
    torch = _torch_imports()
    from transformers import AutoTokenizer

    configs = configs if configs is not None else default_configs()
    config_data = load_config()
    tokenizer_id = tokenizer_id or config_data.get("tokenization", {}).get(
        "tokenizer_id", DEFAULT_TOKENIZER_ID
    )

    records = load_train_records()
    if not records:
        raise FileNotFoundError(
            f"No training records at {TRAIN_PATH} — the benchmark needs a "
            "read-only sample of the real dataset to be meaningful."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=hf_token)
    results: list[dict[str, Any]] = []
    for config in configs:
        sample, sample_report = build_length_sample(
            records, tokenizer, config.max_seq_length, count=sample_count, seed=config.seed
        )
        if verbose:
            print(
                f"\n=== {config.label()}: sample {sample_report.selected} "
                f"({sample_report.selection}), "
                f"sample lengths {sample_report.token_lengths['sample_min']}-"
                f"{sample_report.token_lengths['sample_max']} ==="
            )
        outcome = run_single_benchmark(
            config, tokenizer, sample, model_id=model_id, hf_token=hf_token, verbose=verbose
        )
        outcome["sample_report"] = sample_report.to_dict()
        results.append(outcome)

    return {
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "environment": environment_info(),
        "results": results,
        "dataset_read_only": True,
        "note": (
            "Read-only benchmark: a handful of optimization steps per config; "
            "no dataset file was modified and no training run was performed."
        ),
    }


# ---------------------------------------------------------------------------
# Renderers (pure — unit-testable)
# ---------------------------------------------------------------------------


def render_result_summary(result: dict[str, Any]) -> str:
    """One-configuration summary block in the requested format."""
    config = result["config"]
    steps = result["steps"]
    env = result.get("environment", {})
    lines = [
        f"Sequence length: {config['max_seq_length']}",
        f"Batch size: {config['batch_size']}",
        f"Quantization: {config.get('quantization', '4-bit')}",
        f"Gradient checkpointing: {'enabled' if config.get('gradient_checkpointing') else 'disabled'}",
        f"Peak VRAM: {steps.get('peak_vram_gb')} GB",
        f"CUDA OOM: {'YES' if steps.get('cuda_oom') else 'NO'}",
        f"Forward pass: {steps.get('forward')}",
        f"Backward pass: {steps.get('backward')}",
        f"Training step: {steps.get('training_step')}",
        f"Approximate step time: {steps.get('step_time_seconds')} seconds",
    ]
    if env.get("gpu_name"):
        lines.append(f"GPU: {env['gpu_name']} ({env.get('gpu_vram_gb')} GB)")
    for note in steps.get("notes", []):
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def render_results_table(results: list[dict[str, Any]]) -> str:
    """Markdown comparison table across all configurations."""
    header = (
        "| Seq Length | Batch | Packing | Peak VRAM (GB) | OOM | "
        "Forward | Backward | Training Step | Step Time (s) |\n"
        "|---:|---:|---|---:|---|---|---|---|---:|"
    )
    rows = []
    for result in results:
        config = result["config"]
        steps = result["steps"]
        peak = steps.get("peak_vram_gb")
        rows.append(
            f"| {config['max_seq_length']} | {config['batch_size']} | "
            f"{'yes' if config.get('packing') else 'no'} | "
            f"{peak if peak is not None else 'n/a'} | "
            f"{'YES' if steps.get('cuda_oom') else 'NO'} | "
            f"{steps.get('forward')} | {steps.get('backward')} | "
            f"{steps.get('training_step')} | "
            f"{steps.get('step_time_seconds') if steps.get('step_time_seconds') is not None else 'n/a'} |"
        )
    return "\n".join([header, *rows])


def save_benchmark_report(report: dict[str, Any], path: Path | None = None) -> Path:
    """Persist the full machine-readable report next to the repo outputs."""
    path = path or PROJECT_ROOT / "outputs" / "logs" / "qlora_vram_benchmark_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (requires CUDA; the notebook cells call run_benchmark)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="QLoRA VRAM feasibility benchmark (GPU required)."
    )
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--json", action="store_true", help="print the raw JSON report")
    args = parser.parse_args(argv)

    problems = preflight_errors()
    if problems:
        for problem in problems:
            print(f"preflight: {problem}", file=__import__("sys").stderr)
        return 1

    from src.data.utils import get_hf_token

    configs = [
        BenchmarkConfig(max_seq_length=length, batch_size=args.batch_size)
        for length in args.seq_lengths
    ]
    report = run_benchmark(
        configs=configs,
        model_id=args.model,
        hf_token=get_hf_token(),
        sample_count=args.sample_count,
    )
    report_path = save_benchmark_report(report)
    for result in report["results"]:
        print(render_result_summary(result))
        print()
    print(render_results_table(report["results"]))
    print(f"\nbenchmark: full report saved to {report_path}")
    return 0 if all(r["ok"] for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
