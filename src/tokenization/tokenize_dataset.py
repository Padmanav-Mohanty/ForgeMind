"""Tokenize the final ForgeMind dataset and analyze sequence-length caps.

Loads ONLY the tokenizer (no model weights), renders every example through
the tokenizer's chat template — the same rendering the TRL fine-tuning stage
will use — measures token lengths, and evaluates the candidate
`max_seq_length` caps from configs/data_config.yaml.

Read-only: the input JSONL is never modified, and no example is truncated
before measurement. Outputs:
  - data/final/token_length_report.json  (measured statistics + cap analysis)
  - data/final/long_examples.json        (details for over-cap examples)
The human-readable report is printed to stdout (--json for the raw report).

Usage:
    python -m src.tokenization.tokenize_dataset                 # all splits
    python -m src.tokenization.tokenize_dataset --split train   # one split
    python -m src.tokenization.tokenize_dataset --tokenizer other/model
    scripts/analyze_tokens.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.data.utils import PROJECT_ROOT, load_config
from src.tokenization.render import (
    completion_token_span,
    count_tokens,
    extract_conversation,
    percentile,
)

FINAL_DIR = PROJECT_ROOT / "data" / "final"
SPLIT_FILES = ("train", "validation", "test")

DEFAULT_TOKENIZER_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_CANDIDATE_CAPS = (1024, 2048, 4096)


def _read_split(path: Path) -> list[dict[str, Any]]:
    """Read a final split without failing on one malformed line."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # keep a placeholder so indexes stay aligned with the file;
                # the validator reports malformed lines in detail.
                records.append({"_malformed_line": line_number})
    return records


def _example_identifiers(example: dict[str, Any]) -> dict[str, Any]:
    """Best stable identifiers for a long-example listing."""
    metadata = example.get("metadata") or {}
    fields = [
        "source", "task", "project", "group_key", "bug_id",
        "issue_number", "pull_number", "issue_url", "report_url",
    ]
    return {field: metadata[field] for field in fields if metadata.get(field)}


def _summarize_lengths(lengths: list[int]) -> dict[str, Any]:
    """Summary statistics over a list of per-example token lengths."""
    if not lengths:
        return {"examples": 0}
    ordered = sorted(lengths)
    n = len(ordered)
    total = sum(ordered)
    return {
        "examples": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(total / n, 1),
        "median": percentile(ordered, 50),
        "p75": percentile(ordered, 75),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "total_tokens": total,
        "avg_tokens_per_example": round(total / n, 1),
    }


def _histogram(lengths: list[int], bucket_size: int = 256) -> dict[str, int]:
    """Token-length histogram over fixed-width buckets, e.g. '0-255'."""
    histogram: dict[str, int] = {}
    for length in lengths:
        bucket = (length // bucket_size) * bucket_size
        key = f"{bucket}-{bucket + bucket_size - 1}"
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda kv: int(kv[0].split("-")[0])))


def _cap_analysis(
    lengths: list[int], candidate_caps: list[int]
) -> list[dict[str, Any]]:
    """For each candidate cap: how many examples/tokens it would affect."""
    if not lengths:
        return []
    ordered = sorted(lengths)
    total_examples = len(ordered)
    total_tokens = sum(ordered)

    rows: list[dict[str, Any]] = []
    for cap in sorted(candidate_caps):
        # first index whose length exceeds the cap (binary search)
        low, high = 0, len(ordered)
        while low < high:
            mid = (low + high) // 2
            if ordered[mid] > cap:
                high = mid
            else:
                low = mid + 1
        exceeding = total_examples - low
        affected_tokens = sum(ordered[low:]) if exceeding else 0
        rows.append(
            {
                "max_seq_length": cap,
                "examples_within": low,
                "examples_exceeding": exceeding,
                "percent_examples_exceeding": round(100 * exceeding / total_examples, 2),
                "tokens_within_cap": total_tokens - affected_tokens,
                "tokens_affected": affected_tokens,
                "percent_tokens_affected": round(
                    100 * affected_tokens / total_tokens, 2
                )
                if total_tokens
                else 0.0,
            }
        )
    return rows


def _load_tokenizer(tokenizer_id: str) -> Any:
    """Load only the tokenizer for tokenizer_id (never model weights)."""
    from transformers import AutoTokenizer

    from src.data.utils import get_hf_token

    return AutoTokenizer.from_pretrained(tokenizer_id, token=get_hf_token())


def analyze_split(
    records: list[dict[str, Any]],
    tokenizer: Any,
    candidate_caps: list[int],
    long_examples_limit: int,
) -> dict[str, Any]:
    """Measure one split: statistics, histogram, cap analysis, long examples.

    Over-cap example details are collected for the smallest candidate cap
    (the superset — every example exceeding a larger cap also exceeds it).
    """
    lengths: list[int] = []
    completion_lengths: list[int] = []
    unrenderable: list[dict[str, Any]] = []
    long_examples: list[dict[str, Any]] = []
    over_cap_count = 0
    smallest_cap = min(candidate_caps) if candidate_caps else 0

    for index, example in enumerate(records):
        conversation = extract_conversation(example)
        if conversation is None:
            unrenderable.append({"index": index, **_example_identifiers(example)})
            continue
        length = count_tokens(conversation, tokenizer)
        lengths.append(length)
        completion_lengths.append(completion_token_span(conversation, tokenizer))
        if smallest_cap and length > smallest_cap:
            over_cap_count += 1
            if len(long_examples) < long_examples_limit:
                long_examples.append(
                    {"index": index, "token_length": length, **_example_identifiers(example)}
                )

    return {
        "statistics": _summarize_lengths(lengths),
        "prompt_completion": {
            "avg_completion_tokens": round(
                sum(completion_lengths) / len(completion_lengths), 1
            )
            if completion_lengths
            else 0,
            "completion_share_percent": round(
                100 * sum(completion_lengths) / max(sum(lengths), 1), 1
            )
            if lengths
            else 0.0,
        },
        "histogram": _histogram(lengths),
        "cap_analysis": _cap_analysis(lengths, candidate_caps),
        "unrenderable_examples": unrenderable,
        "rendered_examples": len(lengths),
        "long_examples": {
            "max_seq_length": smallest_cap,
            "listed": long_examples,
            "total_over_cap": over_cap_count,
            "listing_truncated": over_cap_count > len(long_examples),
        },
    }


def analyze_dataset(
    tokenizer: Any,
    config: dict[str, Any],
    splits: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze the configured final splits and assemble the report."""
    tokenization_cfg = config.get("tokenization", {})
    candidate_caps = [
        int(c)
        for c in tokenization_cfg.get("candidate_max_seq_lengths", list(DEFAULT_CANDIDATE_CAPS))
    ]
    long_examples_limit = int(tokenization_cfg.get("long_examples_report_limit", 50))
    splits = splits or list(SPLIT_FILES)

    report: dict[str, Any] = {
        "tokenizer_id": getattr(tokenizer, "name_or_path", ""),
        "counting_method": (
            "len(tokenizer.apply_chat_template(messages, tokenize=True, "
            "add_generation_prompt=False)) — BOS + the tokenizer's chat "
            "template rendering (role framing, default dated system preamble, "
            "content, one <|eot_id|> per message). Matches the TRL "
            "fine-tuning rendering; nothing is truncated before measurement."
        ),
        "candidate_max_seq_lengths": candidate_caps,
        "splits": {},
    }

    for split in splits:
        path = FINAL_DIR / f"{split}.jsonl"
        if not path.is_file():
            report["splits"][split] = {"error": f"{path.name} not found"}
            continue
        records = _read_split(path)
        report["splits"][split] = analyze_split(
            records, tokenizer, candidate_caps, long_examples_limit
        )
        print(
            f"tokenize_dataset: {split}: {len(records)} examples, "
            f"{report['splits'][split]['statistics'].get('total_tokens', 0)} tokens"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Token-length analysis for the ForgeMind final dataset."
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_FILES,
        action="append",
        help="split to analyze (repeatable; default: all)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer id override (default: tokenization.tokenizer_id from config)",
    )
    parser.add_argument("--json", action="store_true", help="print the raw JSON report")
    args = parser.parse_args(argv)

    config = load_config()
    tokenizer_id = args.tokenizer or config.get("tokenization", {}).get(
        "tokenizer_id", DEFAULT_TOKENIZER_ID
    )

    try:
        tokenizer = _load_tokenizer(tokenizer_id)
    except Exception as exc:  # noqa: BLE001 - report and fail cleanly
        print(
            f"tokenize_dataset: failed to load tokenizer '{tokenizer_id}': {exc}",
            file=sys.stderr,
        )
        return 1

    report = analyze_dataset(tokenizer, config, splits=args.split)

    report_path = FINAL_DIR / "token_length_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Long-example details, split out so the full listing can be reviewed
    # without scrolling through the aggregate report.
    long_examples: dict[str, Any] = {
        split: analysis["long_examples"]
        for split, analysis in report["splits"].items()
        if "long_examples" in analysis
    }
    long_path = FINAL_DIR / "long_examples.json"
    long_path.write_text(json.dumps(long_examples, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report, tokenizer_id)

    print(f"\ntokenize_dataset: report saved to {report_path}")
    print(f"tokenize_dataset: long-example listing saved to {long_path}")
    return 0


def _print_report(report: dict[str, Any], tokenizer_id: str) -> None:
    """Human-readable rendering of the token-length report."""
    print("ForgeMind Token-Length Analysis")
    print("===============================")
    print(f"Tokenizer: {tokenizer_id} (tokenizer only — no model weights loaded)")
    print("Counting method: len(apply_chat_template(messages, tokenize=True)),")
    print("  BOS + chat-template framing included — identical to the TRL")
    print("  fine-tuning rendering. Nothing truncated before measurement.")
    for split, analysis in report["splits"].items():
        print(f"\n--- {split} " + "-" * (58 - len(split)))
        if "error" in analysis:
            print(f"  {analysis['error']}")
            continue
        stats = analysis["statistics"]
        if not stats.get("examples"):
            print("  no renderable examples")
            continue
        print(
            f"  examples: {stats['examples']} "
            f"(unrenderable: {len(analysis['unrenderable_examples'])})"
        )
        print(f"  min: {stats['min']}  max: {stats['max']}")
        print(f"  mean: {stats['mean']}  median: {stats['median']}")
        print(
            f"  p75: {stats['p75']}  p90: {stats['p90']}  "
            f"p95: {stats['p95']}  p99: {stats['p99']}"
        )
        print(
            f"  total tokens: {stats['total_tokens']}  "
            f"avg/example: {stats['avg_tokens_per_example']}"
        )
        pc = analysis["prompt_completion"]
        print(
            f"  avg completion tokens: {pc['avg_completion_tokens']} "
            f"({pc['completion_share_percent']}% of tokens)"
        )
        print("  histogram (256-token buckets):")
        for bucket, count in analysis["histogram"].items():
            bar = "#" * max(1, round(40 * count / max(analysis["histogram"].values())))
            print(f"    {bucket:>9}: {count:>5}  {bar}")
        print("  cap analysis:")
        print(
            "    Max Length | Examples Exceeding | % Examples | "
            "Tokens Affected | % Tokens"
        )
        print(
            "    -----------+--------------------+------------+"
            "-----------------+---------"
        )
        for row in analysis["cap_analysis"]:
            print(
                f"    {row['max_seq_length']:>10} | "
                f"{row['examples_exceeding']:>18} | "
                f"{row['percent_examples_exceeding']:>9.2f}% | "
                f"{row['tokens_affected']:>15} | "
                f"{row['percent_tokens_affected']:>7.2f}%"
            )


if __name__ == "__main__":
    raise SystemExit(main())
