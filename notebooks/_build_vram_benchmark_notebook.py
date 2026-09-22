"""Generate notebooks/ForgeMind_QLoRA_VRAM_Benchmark.ipynb.

Kept in the repository so the notebook can be regenerated after edits to
the cell sources (run from the repository root):

    python notebooks/_build_vram_benchmark_notebook.py

The notebook itself delegates all benchmark logic to src/training/benchmark.py;
these cells only bootstrap the environment, orchestrate the calls, and
render the report.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "ForgeMind_QLoRA_VRAM_Benchmark.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    md("""# ForgeMind QLoRA VRAM Feasibility Benchmark (Llama 3.2 3B Instruct)

**Purpose:** measure — not guess — whether the intended 4-bit QLoRA setup can
train at `max_seq_length` **2048** and **4096** on this GPU (target: Colab **T4 16 GB**).

- Model: `meta-llama/Llama-3.2-3B-Instruct` (the exact ForgeMind target model)
- Stack: Transformers + PEFT + bitsandbytes (NF4 4-bit) + LoRA, gradient checkpointing on
- Data: a tiny **read-only** representative sample from `data/final/train.jsonl`,
  rendered through the **real Llama 3.2 chat template** (the same counting
  convention as `src/tokenization/`)
- **Read-only benchmark:** a handful of optimization steps per configuration.
  No dataset file is modified, nothing is written into `data/`, and no real
  training run is started.

**Runtime requirement:** NVIDIA T4 16 GB (Runtime → Change runtime type → T4 GPU).
Run the cells **top to bottom, in order**."""),
    code("""# Cell 1 — Bootstrap: locate the ForgeMind repository
import os, sys
from pathlib import Path

CANDIDATES = [
    Path("/content/ForgeMind"),                # git-cloned into Colab
    Path(os.environ.get("FORGEMIND_ROOT", "")),  # explicit override
    Path.cwd(),                                # running from inside the repo
]

REPO_ROOT = next(
    (p.resolve() for p in CANDIDATES if p and (p / "src" / "training" / "benchmark.py").is_file()),
    None,
)

if REPO_ROOT is None:
    raise RuntimeError(
        "ForgeMind repository not found.\\n"
        "Option A (recommended):  !git clone https://github.com/Padmanav-Mohanty/ForgeMind.git /content/ForgeMind\\n"
        "Option B: upload the repository zip and extract it to /content/ForgeMind\\n"
        "Option C: mount Drive and set FORGEMIND_ROOT to the repo path, then re-run."
    )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.data.utils import PROJECT_ROOT  # noqa: E402  (repo's own root resolution)

assert (PROJECT_ROOT / "src" / "training" / "benchmark.py").is_file()
TRAIN_OK = (PROJECT_ROOT / "data" / "final" / "train.jsonl").is_file()
print(f"Repository root: {PROJECT_ROOT}")
print(f"Train split available: {TRAIN_OK}")
if not TRAIN_OK:
    print("WARNING: data/final/train.jsonl is missing — the benchmark needs a "
          "read-only sample of the real dataset. Upload it before continuing.")"""),
    md("""## Install the training stack

Colab already ships a CUDA-matched PyTorch. **Only** install the pieces on
top of it — reinstalling torch can pull a wheel mismatched with Colab's
CUDA/driver and break `bitsandbytes`."""),
    code("""# Cell 2 — Install (safe on stock Colab; do NOT reinstall torch here)
%pip install -q -U transformers peft trl bitsandbytes accelerate"""),
    code("""# Cell 3 — Environment + GPU verification (fails gracefully, never crashes)
import json

from src.training.benchmark import environment_info, preflight_errors

info = environment_info()
print(json.dumps(info, indent=2, default=str))

if info.get("nvidia_smi"):
    print("\\n--- nvidia-smi ---")
    import subprocess
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

problems = preflight_errors()
if problems:
    print("\\nPREFLIGHT PROBLEMS — fix before running the benchmark:")
    for problem in problems:
        print(f"  - {problem}")
    raise RuntimeError("Preflight failed; see the messages above.")
print("\\nPreflight OK: CUDA + bitsandbytes ready, train split found.")"""),
    md("""## Hugging Face authentication

`meta-llama/Llama-3.2-3B-Instruct` is a **gated** model:

1. Sign in on huggingface.co and accept the license on the
   [model page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
2. Provide a token (Settings → Access Tokens) via **either**:
   - a Colab Secret named `HF_TOKEN` (key icon on the left), **or**
   - running `huggingface-cli login` / `notebook_login()` in a scratch cell.

The repo's own `.env`/`get_hf_token()` path also works when the repository
(including `.env`) is uploaded."""),
    code("""# Cell 4 — Resolve a Hugging Face token (never printed)
import os

from src.data.utils import get_hf_token

HF_TOKEN = None
try:
    from google.colab import userdata  # type: ignore
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
        print("Using HF_TOKEN from Colab Secrets.")
    except Exception:
        pass
except ImportError:
    pass

if HF_TOKEN is None:
    HF_TOKEN = get_hf_token()
    if HF_TOKEN:
        print("Using HF_TOKEN from the repository .env / environment.")
    else:
        print("No token found. If the model download is refused (401/403), "
              "run `from huggingface_hub import notebook_login; notebook_login()` "
              "in a scratch cell, then re-run this cell.")"""),
    md("""## Benchmark configuration

Defaults mirror the intended QLoRA setup: NF4 4-bit + double quant, LoRA
r=16/alpha=32 on the attention projections, gradient checkpointing on,
`per_device_train_batch_size=1`. **Gradient accumulation (4) is recorded for
context only — it does not change per-device VRAM.** Each configuration gets
a fresh model load, and CUDA memory is reset before and after, so peaks are
comparable and one OOM cannot contaminate the next test."""),
    code("""# Cell 5 — Shared imports and the benchmark matrix
from src.training.benchmark import (
    BenchmarkConfig,
    configure_padding,
    default_configs,
    focused_full_step_config,
    load_train_records,
    optional_batch2_configs,
    render_result_summary,
    render_results_table,
    render_sweep_table,
    run_benchmark,
    run_sweep,
    save_benchmark_report,
)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
RESULTS = {}   # label -> report, accumulated across cells

CONFIGS_2048 = [BenchmarkConfig(max_seq_length=2048)]
CONFIGS_4096 = [BenchmarkConfig(max_seq_length=4096)]
print("Benchmark matrix:")
for config in default_configs():
    print(f"  - {config.label()} (grad-accum {config.gradient_accumulation_steps}, "
          f"packing {config.packing}, LoRA r{config.lora_r})")"""),
    md("""### Test A — max_seq_length = 2048

Loads the 4-bit model fresh, samples 4 representative near-cap examples
(seeded, read-only), then measures forward → backward → optimizer steps.
OOM is caught per phase and reported, not raised."""),
    code("""# Cell 6 — Run the 2048 benchmark
report_2048 = run_benchmark(
    configs=CONFIGS_2048, model_id=MODEL_ID, hf_token=HF_TOKEN, verbose=True
)
for result in report_2048["results"]:
    label = result["config"]["label"]
    RESULTS[label] = result
    print("\\n" + render_result_summary(result))"""),
    md("""### Test B — max_seq_length = 4096

Same procedure, fresh model load, CUDA cache cleared between tests."""),
    code("""# Cell 7 — Run the 4096 benchmark
report_4096 = run_benchmark(
    configs=CONFIGS_4096, model_id=MODEL_ID, hf_token=HF_TOKEN, verbose=True
)
for result in report_4096["results"]:
    label = result["config"]["label"]
    RESULTS[label] = result
    print("\\n" + render_result_summary(result))"""),
    md("""## Results"""),
    code("""# Cell 8 — Comparison table + persisted report
report_path = save_benchmark_report(
    {
        "model_id": MODEL_ID,
        "padding": report_4096.get("padding"),
        "environment": report_4096["environment"],
        "results": list(RESULTS.values()),
    }
)

print(render_results_table(list(RESULTS.values())))
print(f"\\nEnvironment: {report_4096['environment'].get('gpu_name')} / "
      f"{report_4096['environment'].get('gpu_vram_gb')} GB / "
      f"torch {report_4096['environment'].get('torch')} / "
      f"transformers {report_4096['environment'].get('transformers')} / "
      f"peft {report_4096['environment'].get('peft')} / "
      f"bitsandbytes {report_4096['environment'].get('bitsandbytes')}")
print(f"\\nFull JSON report: {report_path}")
print("Copy the summaries + table above back for the cap decision.")"""),
    md("""### Sequence-length feasibility sweep — 512 / 768 / 1024 / 1280 / 1536

After the primary matrix OOMed during **forward** at 2048 and 4096, the open
question shifts to the largest sequence length whose forward pass fits at
all. This sweep runs the **same forward-pass test** as the primary benchmark
at five shorter lengths — every other variable held identical: batch size 1,
packing off, gradient checkpointing on, the same 4-bit NF4 quantization and
LoRA r16/α32 configuration, and the same seeded near-cap/mid sample
methodology (tokenized once and reused across all lengths).

**Forward-only by design:** no backward pass and no optimizer step execute,
so each row isolates the activation footprint — the variable the sweep is
actually probing. VRAM is reported **only when the forward phase completes**;
a failed run shows `n/a`, never the model-load peak. The sweep is additive:
the 2048/4096 results above remain valid, and the two tables read together
give the full feasibility picture. Finding OOM lengths is a *successful*
sweep — the boundary between the largest PASS and the first OOM is the
measurement."""),
    code("""# Cell 9 — Forward-pass feasibility sweep (512-1536)
SWEEP_LENGTHS = (512, 768, 1024, 1280, 1536)

sweep_report = run_sweep(
    seq_lengths=SWEEP_LENGTHS, model_id=MODEL_ID, hf_token=HF_TOKEN, verbose=True
)
for result in sweep_report["results"]:
    RESULTS[result["config"]["label"]] = result
    print(render_result_summary(result))
print("\\n" + render_sweep_table(sweep_report["results"]))
print("\\nFull sweep JSON:", save_benchmark_report(
    sweep_report,
    path=Path("outputs/logs/qlora_vram_sweep_report.json"),
))"""),
    md("""### Focused probe — full training step at seq 1024

The sweep measured forward **PASS** through 1024 and OOM from 1280 up, but a
forward pass fitting is necessary, not sufficient: backward needs gradient
workspace plus re-computed activations, and the optimizer step adds adapter
optimizer state. This probe runs the **complete forward → backward →
optimizer-step sequence** at 1024 with every other variable identical —
batch 1, packing off, gradient checkpointing on, the same 4-bit NF4
quantization and LoRA r16/α32 — through the exact runner and seeded sample
methodology of the primary benchmark. Each phase reports separately; peak
VRAM is printed **only when all three phases complete** — a backward or
optimizer OOM is reported as such, never masked. It refines the sweep's 1024
row into the decision number."""),
    code("""# Cell 10 — Focused full-step probe: seq 1024, forward + backward + optimizer
probe_report = run_benchmark(
    configs=[focused_full_step_config()],
    model_id=MODEL_ID, hf_token=HF_TOKEN, verbose=True,
)
for result in probe_report["results"]:
    RESULTS[result["config"]["label"]] = result
    print(render_result_summary(result))
print("\\nProbe JSON:", save_benchmark_report(
    probe_report,
    path=Path("outputs/logs/qlora_vram_focused_1024_report.json"),
))"""),
    md("""### Optional follow-ups (run only if the batch-1 tests passed)

**Batch size 2** at 2048 — secondary question; the primary one is sequence
feasibility. Set `RUN_OPTIONAL = True` to execute."""),
    code("""# Cell 11 (optional) — batch size 2 at 2048
RUN_OPTIONAL = False  # set True to run

if RUN_OPTIONAL:
    report_bs2 = run_benchmark(
        configs=optional_batch2_configs(), model_id=MODEL_ID, hf_token=HF_TOKEN
    )
    for result in report_bs2["results"]:
        RESULTS[result["config"]["label"]] = result
        print(render_result_summary(result))
else:
    print("Skipped (RUN_OPTIONAL = False).")"""),
    md("""**Packing — test it?** ForgeMind's training configuration does **not** set
packing today (`configs/qlora.yaml` defines only `max_seq_length`; TRL SFT
defaults to `packing=False`), so the measured path above is **non-packing**
and that is the number the cap decision should rely on. Packing concatenates
examples into dense sequences: it changes step-time economics and pad-token
waste, but the worst-case per-sequence activation memory is what the
non-packed run already exercises. The optional cell below measures a
worst-case single **packed** sequence at each cap for completeness — keep it
separate and do not mix its numbers into the non-packed table."""),
    code("""# Cell 12 (optional) — packed worst-case single sequence per cap
RUN_PACKING = False  # set True to run

if RUN_PACKING:
    from src.tokenization.render import extract_conversation
    from src.training.benchmark import (
        BenchmarkConfig, build_length_sample, configure_padding,
        load_train_records, run_single_benchmark,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    configure_padding(tokenizer)  # EOS-as-pad + left padding BEFORE any padding=True call
    records = load_train_records()
    for cap in (2048, 4096):
        sample, _ = build_length_sample(records, tokenizer, cap, count=4)
        # one packed text: real near-cap conversations concatenated up to the cap
        packed, total = [], 0
        for item in sample:
            conversation = extract_conversation({"messages": item["messages"]})
            rendered = tokenizer.apply_chat_template(conversation, tokenize=False)
            packed.append(rendered)
            total += len(tokenizer(rendered)["input_ids"])
            if total >= cap * 0.98:
                break
        config = BenchmarkConfig(max_seq_length=cap, packing=True)
        result = run_single_benchmark(
            config, tokenizer, sample, model_id=MODEL_ID, hf_token=HF_TOKEN,
            texts_override=["\\n".join(packed)],
        )
        RESULTS[config.label()] = result
        print(render_result_summary(result))
else:
    print("Skipped (RUN_PACKING = False).")"""),
    md("""## Interpretation guide

- **Peak VRAM** is `torch.cuda.max_memory_allocated` over the whole
  configuration — model weights (4-bit ≈ ~2 GB for 3B), LoRA + optimizer
  state, and the largest activation footprint. Compare it against the T4's
  ~15.6 GB *usable*; leave headroom for fragmentation (allocator spikes above
  the measured peak are normal on long sequences). It is reported **only when
  all three phases (forward, backward, optimizer step) PASS**; otherwise it
  is withheld (`n/a`) — the counter at that point merely reflects the model
  weights loading (~3.5 GB), which is not a training measurement.
- **Padding** is configured before batching: Llama 3.2 ships no pad token,
  so the tokenizer reuses EOS (`pad_token = eos_token`, no new vocabulary
  entry) and pads on the **left**. Labels are masked via the attention mask,
  not by token id, so the template's real `<|eot_id|>` turn separators stay
  inside the loss. The report records this under its top-level `padding` key.
- **OOM at a phase** says where the ceiling is: forward/backward OOM at 4096
  but PASS at 2048 means the cap decision is memory-bound; OOM only at the
  optimizer step means optimizer state is the marginal cost.
- **Step time** is the average over the timed optimizer steps after warmup.
  Rough epoch estimate for non-packed training:
  `steps ≈ ceil(2816 / (batch × grad_accum))`; `epoch_time ≈ steps × step_time × (count of accumulation micro-steps)`.
  Use the measured token totals (train ≈ 4.0 M tokens) rather than assumptions.
- **Batch size 1 is the decision-grade number.** Gradient accumulation
  changes effective batch size and wall-clock, not per-device VRAM — don't
  conflate them.
- **Reading the sweep (512-1536):** rows are forward-pass-only, so their
  Peak VRAM reflects 4-bit weights + LoRA + activations at that length —
  subtract a sweep row's VRAM from the T4's usable ~15.6 GB to see how much
  room remains for backward-pass workspace (gradients, checkpoint
  re-computation, optimizer state). A length whose forward fits but whose
  full benchmark OOMed in backward has no headroom for training at that
  length. The useful boundary is the largest sweep length that PASSes —
  lengths below it are comfortable, lengths above it do not fit even a
  forward pass.
- **The focused 1024 probe is the decision number.** It converts the
  sweep's forward-only boundary into the full-step verdict at that length:
  all three phases PASS means 1024 (with over-cap trimming) is the only cap
  the T4 supports as-is; a backward OOM means even 1024 cannot *train* under
  the intended stack, and the stack assumptions (not the cap) need revisiting.
- These results fill the last gap before setting `max_seq_length` in
  `configs/qlora.yaml`. The dataset-side facts already measured: train
  mean ≈ 1,423 tokens, p95 ≈ 3,255, p99 ≈ 5,210, max ≈ 25,642; at 2048,
  681/2,816 examples exceed the cap (581 rescuable by user-side evidence
  trimming, 99 would need completion-side changes, 1 not rescuable); at 4096,
  71 exceed (70 rescuable). Context only — the benchmark does not act on it.

### Known dataset artifact (documented, not touched here)

`src/data/ghpr.py` (lines ~267-270) builds the assistant's `### Issue Summary`
as `issue_body.split("\\n")[0]` — the **entire first line of the issue body**.
For train idx 1830 (`ghpr:tikv/tikv#3334`) that line is a multi-kilobyte log
line, so it leaked into the completion (4,616 completion tokens). The
benchmark's sample builder may pick this example; its memory profile is still
valid (it *is* what the current dataset contains), but the example itself
should be fixed at the converter level in a separate, deliberate change.

### What this benchmark did NOT do

No dataset file was created, modified, filtered, or trimmed; `max_seq_length`
in `configs/qlora.yaml` remains unset; no training run, checkpoint, or
evaluation was performed."""),
    code("""# Cell 13 — Final cleanup
import gc, torch

RESULTS.clear()
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(0)
print("CUDA cache cleared; benchmark session cleaned up.")"""),
]


def build() -> Path:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return NB_PATH


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({len(CELLS)} cells)")
