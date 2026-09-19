"""Clean processed examples: quality filters and malformed-record removal.

Phase 5 of the pipeline. The cleaner NEVER edits raw data; it only decides
which processed examples survive, and reports statistics.

Removal rules (order-independent, all deterministic):
  1. structurally invalid: not a dict, missing/empty `messages`
  2. missing user or assistant content (both roles must exist, non-empty)
  3. invalid roles (anything outside system/user/assistant)
  4. placeholder junk: TODO/FIXME/XXX/lorem-style bodies, template
     artifacts like "<insert ...>", "...", "N/A" as whole content
  5. too short by configured minimums -- with a code exemption: examples
     containing fenced/indented code get lower minimums, because a short
     bug-fix snippet is still a valid training signal. Short-but-valid
     examples are NOT aggressively deleted.

Statistics are recorded to data/processed/cleaning_report.json.

Usage:
    python -m src.data.clean            # all processed files
    python -m src.data.clean ghpr
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.utils import (
    PROJECT_ROOT,
    contains_code,
    get_assistant_content,
    get_user_content,
    load_config,
    read_jsonl,
    write_jsonl,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"

DATASET_FILES = {
    "aifaultbench": "aifaultbench.jsonl",
    "ghpr": "ghpr.jsonl",
    "defects4j": "defects4j.jsonl",
}

PLACEHOLDER_PATTERNS = [
    r"^\s*(TODO|FIXME|XXX)\s*:?\s*$",
    r"^\s*(lorem ipsum|placeholder|tbd|to be determined|n/?a)\s*[\.\!]?\s*$",
    r"^<insert[^>]*>$",
    r"^\{\{\s*(prompt|input|content|text)\s*\}\}$",
    r"^\s*\.{3,}\s*$",
]


def _is_placeholder(text: str) -> bool:
    """True when the ENTIRE content is placeholder junk.

    Anchors apply to the whole (stripped) text: a long issue body that
    merely contains a line reading "None" or "..." is still a valid
    example and must not be rejected.
    """
    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered in {"...", "…", "-", "--", "n/a", "na", "none", "null", "tbd"}:
        return True
    return any(
        re.search(pattern, stripped, re.IGNORECASE)
        for pattern in PLACEHOLDER_PATTERNS
    )


def _validate_structure(example: Any) -> str | None:
    """Return a rejection reason for structural problems, else None."""
    if not isinstance(example, dict):
        return "not_an_object"
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return "missing_messages"
    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    if any(role not in {"system", "user", "assistant"} for role in roles):
        return "invalid_role"
    if not any(role == "user" for role in roles):
        return "missing_user"
    if not any(role == "assistant" for role in roles):
        return "missing_assistant"
    metadata = example.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return "invalid_metadata"
    return None


def _min_lengths(config: dict, has_code: bool) -> tuple[int, int]:
    cleaning = config.get("cleaning", {})
    if has_code:
        return (
            int(cleaning.get("min_input_length_code", cleaning.get("min_input_length", 20))),
            int(cleaning.get("min_output_length_code", cleaning.get("min_output_length", 20))),
        )
    return int(cleaning.get("min_input_length", 20)), int(cleaning.get("min_output_length", 20))


def clean_file(path: Path, config: dict) -> dict[str, Any]:
    """Clean one processed JSONL file in place; return its statistics."""
    records = read_jsonl(path)
    stats: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []

    for record in records:
        reason = _validate_structure(record)
        if reason is None:
            user_content = get_user_content(record)
            assistant_content = get_assistant_content(record)
            has_code = contains_code(user_content) or contains_code(assistant_content)
            min_in, min_out = _min_lengths(config, has_code)
            if not user_content.strip():
                reason = "empty_user_content"
            elif not assistant_content.strip():
                reason = "empty_assistant_content"
            elif _is_placeholder(user_content) or _is_placeholder(assistant_content):
                reason = "placeholder_content"
            elif len(user_content) < min_in:
                reason = "input_too_short"
            elif len(assistant_content) < min_out:
                reason = "output_too_short"
        if reason is None:
            kept.append(record)
        else:
            stats[reason] += 1

    write_jsonl(path, kept)

    try:
        display_path = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(path)  # e.g. tmp_path in tests

    return {
        "file": display_path,
        "before": len(records),
        "after": len(kept),
        "removed": len(records) - len(kept),
        "removal_reasons": dict(stats),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = load_config()
    targets = argv or list(DATASET_FILES)

    report: dict[str, Any] = {}
    failures = 0
    for name in targets:
        filename = DATASET_FILES.get(name)
        if filename is None:
            print(f"clean: unknown dataset '{name}'", file=sys.stderr)
            failures += 1
            continue
        path = PROCESSED_DIR / filename
        if not path.is_file():
            print(f"clean: {path} not found (convert first?)", file=sys.stderr)
            failures += 1
            continue
        stats = clean_file(path, config)
        report[name] = stats
        print(
            f"clean: {name}: {stats['before']} -> {stats['after']} "
            f"(removed {stats['removed']})"
        )
        for reason, count in sorted(stats["removal_reasons"].items()):
            print(f"    - {reason}: {count}")

    existing: dict[str, Any] = {}
    if REPORT_PATH.is_file():
        try:
            existing = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"clean: report saved to {REPORT_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
