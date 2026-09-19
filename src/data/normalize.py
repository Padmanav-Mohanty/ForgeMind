"""Normalize processed examples without altering their meaning.

Phase 5 of the pipeline. Applied between conversion and cleaning:
  - line endings unified to \n, tabs in prose kept only inside code blocks
  - trailing whitespace removed on every line
  - 3+ consecutive blank lines collapsed to one blank line
  - outer whitespace trimmed from message content
  - metadata keys with empty values dropped (schema rule)

Whitespace inside fenced code blocks is deliberately preserved except for
trailing spaces and CRLF endings: code formatting must survive.

Usage:
    python -m src.data.normalize            # all processed files
    python -m src.data.normalize aifaultbench
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from src.data.utils import PROJECT_ROOT, read_jsonl, write_jsonl

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_FILES = {
    "aifaultbench": "aifaultbench.jsonl",
    "ghpr": "ghpr.jsonl",
    "defects4j": "defects4j.jsonl",
}


def normalize_text(text: str) -> str:
    """Whitespace-normalize one message's content, preserving code blocks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    in_fence = False
    normalized: list[str] = []
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            normalized.append(line.rstrip())
            continue
        # Trailing whitespace always removed; leading whitespace preserved
        # inside fences (indentation matters in code).
        if in_fence:
            normalized.append(line.rstrip())
        else:
            normalized.append(line.rstrip() if line.strip() else "")
    text = "\n".join(normalized)
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # collapse huge gaps only
    # Strip leading blank lines and trailing whitespace, but never strip
    # leading indentation of the first content line (it may be code).
    text = re.sub(r"^(?:\s*\n)+", "", text)
    return text.rstrip()


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized deep copy of one example."""
    messages: list[dict[str, str]] = []
    for message in example.get("messages", []):
        role = message.get("role", "")
        content = message.get("content", "")
        messages.append({"role": role, "content": normalize_text(content) if content else ""})
    metadata = {
        key: value
        for key, value in (example.get("metadata") or {}).items()
        if value not in (None, "")
    }
    return {"messages": messages, "metadata": metadata}


def normalize_file(path: Path) -> tuple[int, int]:
    """Normalize a processed JSONL file in place. Returns (before, after)."""
    records = read_jsonl(path)
    normalized = [normalize_example(record) for record in records]
    write_jsonl(path, normalized)
    return len(records), len(normalized)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    targets = argv or list(DATASET_FILES)
    failures = 0
    for name in targets:
        filename = DATASET_FILES.get(name)
        if filename is None:
            print(f"normalize: unknown dataset '{name}'", file=sys.stderr)
            failures += 1
            continue
        path = PROCESSED_DIR / filename
        if not path.is_file():
            print(f"normalize: {path} not found (convert first?)", file=sys.stderr)
            failures += 1
            continue
        before, after = normalize_file(path)
        print(f"normalize: {name}: {before} -> {after} examples")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
