"""Print ForgeMind dataset statistics.

Phase 8. Reads the processed and final JSONL files (never modifies them)
and prints a summary per source, per split, task, and language, plus
average input/output lengths.

Usage:
    python -m src.data.stats
    python -m src.data.stats --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.utils import PROJECT_ROOT, get_assistant_content, get_user_content, read_jsonl

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
RAW_DIRS = {
    "aifaultbench": PROJECT_ROOT / "data" / "raw" / "aifaultbench",
    "ghpr": PROJECT_ROOT / "data" / "raw" / "ghpr",
    "defects4j": PROJECT_ROOT / "data" / "raw" / "defects4j",
}

PROCESSED_FILES = {
    "aifaultbench": "aifaultbench.jsonl",
    "ghpr": "ghpr.jsonl",
    "defects4j": "defects4j.jsonl",
}

SPLIT_FILES = ("train", "validation", "test")


def _count_raw(name: str) -> int:
    raw_dir = RAW_DIRS[name]
    if not raw_dir.is_dir():
        return 0
    if name == "aifaultbench":
        bugs_dir = raw_dir / "bugs"
        return sum(1 for p in bugs_dir.iterdir() if p.is_dir()) if bugs_dir.is_dir() else 0
    if name == "ghpr":
        csv_path = raw_dir / "dataset" / "ghpr.csv"
        if not csv_path.is_file():
            return 0
        import csv as _csv
        import sys as _sys

        _csv.field_size_limit(min(_sys.maxsize, 2**31 - 1))
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            return max(sum(1 for _ in _csv.DictReader(fh)), 0)  # records, not lines
    if name == "defects4j":
        projects_dir = raw_dir / "framework" / "projects"
        if not projects_dir.is_dir():
            return 0
        total = 0
        for project_dir in projects_dir.iterdir():
            bugs_csv = project_dir / "active-bugs.csv"
            if bugs_csv.is_file():
                with open(bugs_csv, "r", encoding="utf-8", newline="") as fh:
                    total += max(sum(1 for _ in fh) - 1, 0)
        return total
    return 0


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    tasks: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    for record in records:
        metadata = record.get("metadata") or {}
        task = metadata.get("task")
        if task:
            tasks[task] += 1
        language = metadata.get("language")
        if language:
            languages[language] += 1
        source = metadata.get("source")
        if source:
            sources[source] += 1
        input_lengths.append(len(get_user_content(record)))
        output_lengths.append(len(get_assistant_content(record)))
    n = len(records)
    return {
        "examples": n,
        "tasks": dict(sorted(tasks.items(), key=lambda kv: -kv[1])),
        "languages": dict(sorted(languages.items(), key=lambda kv: -kv[1])),
        "sources": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
        "avg_input_chars": round(sum(input_lengths) / n) if n else 0,
        "avg_output_chars": round(sum(output_lengths) / n) if n else 0,
    }


def collect_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {"per_source": {}, "final": {}, "cleaning": {}, "dedup": {}}

    # Load filter reports first so per-source counts can use them.
    for report_name, report_path in (
        ("cleaning", PROCESSED_DIR / "cleaning_report.json"),
        ("dedup", PROCESSED_DIR / "dedup_report.json"),
    ):
        if report_path.is_file():
            try:
                stats[report_name] = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                stats[report_name] = {}

    for name in ("aifaultbench", "ghpr", "defects4j"):
        path = PROCESSED_DIR / PROCESSED_FILES[name]
        processed = read_jsonl(path) if path.is_file() else []
        clean_info = stats["cleaning"].get(name, {})
        stats["per_source"][name] = {
            "raw_examples": _count_raw(name),
            # "converted" = post-normalization count (cleaning 'before'),
            # falling back to the current file count when no report exists.
            "converted_examples": clean_info.get("before", len(processed)),
            "summary": _summarize(processed),
        }

    split_totals = {}
    for split in SPLIT_FILES:
        path = FINAL_DIR / f"{split}.jsonl"
        records = read_jsonl(path) if path.is_file() else []
        split_totals[split] = _summarize(records)
    stats["final"] = {
        "total": sum(s["examples"] for s in split_totals.values()),
        "splits": split_totals,
    }
    return stats


def _fmt_counts(counter: dict[str, int], indent: str = "  ") -> str:
    if not counter:
        return f"{indent}(none)"
    return "\n".join(f"{indent}{key}: {value}" for key, value in counter.items())


def print_stats(stats: dict[str, Any]) -> None:
    print("ForgeMind Dataset Statistics")
    print("============================")
    print()
    display_names = {"aifaultbench": "AIFaultBench", "ghpr": "GHPR", "defects4j": "Defects4J"}
    for name in ("aifaultbench", "ghpr", "defects4j"):
        info = stats["per_source"].get(name, {})
        summary = info.get("summary", {})
        clean_info = stats["cleaning"].get(name, {})
        dedup_info = stats["dedup"].get(name, {})
        print(display_names[name])
        print("-" * len(display_names[name]))
        print(f"Raw examples: {info.get('raw_examples', 0)}")
        print(f"Converted examples: {info.get('converted_examples', 0)}")
        if clean_info:
            print(
                f"After cleaning: {clean_info.get('after', '?')} "
                f"(removed {clean_info.get('removed', '?')})"
            )
        if dedup_info:
            print(
                f"After deduplication: {dedup_info.get('after', '?')} "
                f"(exact {dedup_info.get('exact_removed', '?')}, "
                f"near {dedup_info.get('near_removed', '?')})"
            )
        if summary:
            print(f"Avg input chars: {summary.get('avg_input_chars', 0)}")
            print(f"Avg output chars: {summary.get('avg_output_chars', 0)}")
            print("Tasks:")
            print(_fmt_counts(summary.get("tasks", {})))
            print("Languages:")
            print(_fmt_counts(summary.get("languages", {})))
        print()

    final = stats["final"]
    print("Final Dataset")
    print("-------------")
    print(f"Total: {final.get('total', 0)}")
    for split in SPLIT_FILES:
        info = final.get("splits", {}).get(split, {})
        print(f"{split.capitalize()}: {info.get('examples', 0)}")

    merged: dict[str, Any] = {"tasks": Counter(), "languages": Counter(), "sources": Counter()}
    for split in SPLIT_FILES:
        info = final.get("splits", {}).get(split, {})
        for key in merged:
            for name, count in info.get(key, {}).items():
                merged[key][name] += count
    print()
    print("Task distribution (final):")
    print(_fmt_counts(dict(merged["tasks"].most_common())))
    print()
    print("Language distribution (final):")
    print(_fmt_counts(dict(merged["languages"].most_common())))
    print()
    print("Source distribution (final):")
    print(_fmt_counts(dict(merged["sources"].most_common())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ForgeMind dataset statistics.")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the report")
    args = parser.parse_args(argv)

    stats = collect_stats()
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
