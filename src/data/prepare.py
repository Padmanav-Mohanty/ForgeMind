"""End-to-end data preparation pipeline for ForgeMind.

Runs the full pipeline against already-downloaded raw data:
    convert -> normalize -> clean -> deduplicate -> split

Raw downloading is separate (`python -m src.data.download`) so that raw
acquisition and derived-data generation stay distinct stages.

Usage:
    python -m src.data.prepare                       # all enabled sources
    python -m src.data.prepare --source ghpr         # single source
    python -m src.data.prepare --skip-normalize      # skip whitespace pass
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from src.data.utils import PROJECT_ROOT, load_config

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "pipeline_report.json"

SOURCES = ("aifaultbench", "ghpr", "defects4j")

CONVERTER_MODULES = {
    "aifaultbench": "src.data.aifaultbench",
    "ghpr": "src.data.ghpr",
    "defects4j": "src.data.defects4j",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the ForgeMind final dataset.")
    parser.add_argument(
        "--source",
        choices=SOURCES,
        action="append",
        help="restrict the run to one dataset (repeatable; default: all enabled)",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="skip the whitespace normalization pass",
    )
    args = parser.parse_args(argv)

    config = load_config()
    enabled = [s for s in SOURCES if config.get("datasets", {}).get(s, False)]
    if args.source:
        enabled = [s for s in enabled if s in args.source]
    if not enabled:
        print("prepare: no enabled sources; nothing to do", file=sys.stderr)
        return 1

    failures = 0
    report: dict[str, Any] = {"sources": enabled, "steps": []}

    # 1. Convert raw -> processed ------------------------------------------
    for source in enabled:
        print(f"\n=== convert:{source} " + "=" * 46)
        try:
            module = importlib.import_module(CONVERTER_MODULES[source])
            exit_code = module.convert(config)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"convert:{source} FAILED: {exc}", file=sys.stderr)
            exit_code = 1
        report["steps"].append({"step": f"convert:{source}", "exit_code": exit_code})
        if exit_code != 0:
            failures += 1

    # 2. Normalize ----------------------------------------------------------
    if not args.skip_normalize:
        from src.data.normalize import normalize_file

        for source in enabled:
            path = PROCESSED_DIR / f"{source}.jsonl"
            if not path.is_file():
                continue
            before, after = normalize_file(path)
            print(f"prepare: normalized {source}: {before} -> {after}")
            report["steps"].append({"step": f"normalize:{source}", "before": before, "after": after})

    # 3. Clean ---------------------------------------------------------------
    from src.data.clean import clean_file

    for source in enabled:
        path = PROCESSED_DIR / f"{source}.jsonl"
        if not path.is_file():
            continue
        stats = clean_file(path, config)
        print(f"prepare: cleaned {source}: {stats['before']} -> {stats['after']}")
        report["steps"].append({"step": f"clean:{source}", **stats})

    # 4. Deduplicate ---------------------------------------------------------
    from src.data.deduplicate import deduplicate_file

    for source in enabled:
        path = PROCESSED_DIR / f"{source}.jsonl"
        if not path.is_file():
            continue
        stats = deduplicate_file(path, config)
        print(
            f"prepare: deduplicated {source}: {stats['before']} -> {stats['after']} "
            f"(exact {stats['exact_removed']}, near {stats['near_removed']})"
        )
        report["steps"].append({"step": f"dedup:{source}", **stats})

    # 5. Split ----------------------------------------------------------------
    from src.data.split import split_files

    print("\n=== split " + "=" * 52)
    try:
        split_report = split_files(config, sources=enabled)
    except ValueError as exc:
        print(f"prepare: split failed: {exc}", file=sys.stderr)
        return 1
    report["split"] = split_report
    for split in ("train", "validation", "test"):
        info = split_report["splits"][split]
        print(f"prepare: {split}: {info['examples']} examples ({info['share']:.1%})")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nprepare: pipeline report saved to {REPORT_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
