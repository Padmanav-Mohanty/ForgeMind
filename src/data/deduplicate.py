"""Deduplicate processed examples (exact duplicates + near duplicates).

Phase 6. Fully deterministic — same input always gives the same output.

Exact dedup: SHA-256 over the normalized user+assistant content (whitespace
and case collapsed), so examples that only differ by formatting collapse
into one.

Near dedup: cosine-ish similarity over hashed word-token multisets. For
every example I build a 256-bit MinHash-style signature from 8 hash
projections of its token multiset, and only compare two examples if they
share at least one signature band. The actual decision is Jaccard over the
token multisets, thresholded from config (default 0.85). Kept this stdlib
only — didn't want to pull in a heavy dependency just for this.

Tie-breaking is deterministic too: inside a duplicate cluster, whichever
example appears first in file order is the one that survives, and
survivors keep their original relative order.

Reports before/removed/remaining per file, written out alongside the
cleaning report.

Usage:
    python -m src.data.deduplicate            # all processed files
    python -m src.data.deduplicate ghpr
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
    content_fingerprint,
    get_assistant_content,
    get_user_content,
    jaccard,
    load_config,
    read_jsonl,
    token_set,
    write_jsonl,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "dedup_report.json"

DATASET_FILES = {
    "aifaultbench": "aifaultbench.jsonl",
    "ghpr": "ghpr.jsonl",
    "defects4j": "defects4j.jsonl",
}

# signature params: 8 hash projections, banded into 4 bands of 2 each
NUM_BANDS = 4
BAND_WIDTH = 2
NUM_HASHES = NUM_BANDS * BAND_WIDTH


def _signature_bands(tokens: list[str]) -> list[tuple[int, ...]]:
    """Build MinHash-style band signatures over a token multiset."""
    import hashlib

    bands: list[tuple[int, ...]] = []
    for band in range(NUM_BANDS):
        band_values: list[int] = []
        for h in range(BAND_WIDTH):
            index = band * BAND_WIDTH + h
            values: list[int] = []
            for token in set(tokens):
                digest = hashlib.md5(f"{index}:{token}".encode("utf-8")).digest()
                values.append(int.from_bytes(digest[:8], "big"))
            band_values.append(min(values) if values else 0)
        bands.append(tuple(band_values))
    return bands


def _candidate_pairs(records: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """Find pairs of examples that share at least one signature band."""
    from collections import defaultdict

    bands_by_example: list[list[tuple[int, ...]]] = []
    band_buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)

    for record in records:
        tokens = re.findall(
            r"[a-z0-9_]+",
            (get_user_content(record) + "\n" + get_assistant_content(record)).lower(),
        )
        bands = _signature_bands(tokens)
        bands_by_example.append(bands)
        for band in bands:
            band_buckets[band].append(len(bands_by_example) - 1)

    candidates: set[tuple[int, int]] = set()
    for bucket in band_buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                candidates.add((min(a, b), max(a, b)))
    return candidates


def deduplicate_records(
    records: list[dict[str, Any]],
    near: bool = True,
    near_threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate one dataset's records. Returns (kept, stats)."""
    stats = {"exact_removed": 0, "near_removed": 0}

    # --- exact duplicates first --------------------------------------------
    seen_fingerprints: dict[str, int] = {}
    after_exact: list[dict[str, Any]] = []
    for record in records:
        fingerprint = content_fingerprint(record)
        if fingerprint in seen_fingerprints:
            stats["exact_removed"] += 1
            continue
        seen_fingerprints[fingerprint] = len(after_exact)
        after_exact.append(record)

    if not near:
        return after_exact, stats

    # --- then near duplicates -----------------------------------------------
    token_sets = [token_set(get_user_content(r) + "\n" + get_assistant_content(r)) for r in after_exact]
    candidates = _candidate_pairs(after_exact)

    # union-find over candidate pairs whose Jaccard clears the threshold
    parent = list(range(len(after_exact)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, j in sorted(candidates):
        if jaccard(token_sets[i], token_sets[j]) >= near_threshold:
            union(i, j)

    cluster_representative: dict[int, int] = {}
    kept: list[dict[str, Any]] = []
    for index, record in enumerate(after_exact):
        root = find(index)
        if root in cluster_representative:
            stats["near_removed"] += 1
            continue
        cluster_representative[root] = index
        kept.append(record)

    return kept, stats


def deduplicate_file(path: Path, config: dict) -> dict[str, Any]:
    """Deduplicate one processed JSONL file in place; return stats."""
    records = read_jsonl(path)
    settings = config.get("dedup", {})
    kept, stats = deduplicate_records(
        records,
        near=bool(settings.get("near", True)),
        near_threshold=float(settings.get("near_threshold", 0.85)),
    )
    write_jsonl(path, kept)
    return {
        "file": str(path.relative_to(PROCESSED_DIR)),
        "before": len(records),
        "after": len(kept),
        "removed": len(records) - len(kept),
        "exact_removed": stats["exact_removed"],
        "near_removed": stats["near_removed"],
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
            print(f"deduplicate: unknown dataset '{name}'", file=sys.stderr)
            failures += 1
            continue
        path = PROCESSED_DIR / filename
        if not path.is_file():
            print(f"deduplicate: {path} not found (convert first?)", file=sys.stderr)
            failures += 1
            continue
        stats = deduplicate_file(path, config)
        report[name] = stats
        print(
            f"deduplicate: {name}: {stats['before']} -> {stats['after']} "
            f"(exact {stats['exact_removed']}, near {stats['near_removed']})"
        )

    # merge into whatever's already there so running a subset of datasets
    # doesn't clobber the report entries for the rest
    existing: dict[str, Any] = {}
    if REPORT_PATH.is_file():
        try:
            existing = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"deduplicate: report saved to {REPORT_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())