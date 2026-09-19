"""Split cleaned examples into train/validation/test with leakage control.

Phase 7. Deterministic given the same config (seed) and input files.

Grouping / leakage prevention
-----------------------------
Examples are assigned to splits in whole groups, where a group is
identified by metadata["group_key"]:
  - aifaultbench: one group per fault (bug_id) -- the two derived examples
    (fault analysis + error analysis) always land in the same split
  - ghpr: one group per upstream issue (`ghpr:<slug>#<issue>`) -- all PRs
    linked to the same issue land together
  - defects4j: one group per bug (`defects4j:<project>#<bug_id>`)

Groups (or lone examples) are shuffled with a seeded RNG and dealt into
train/validation/test greedily by target ratios, with the largest
remainder correction. Because groups are atomic, no bug/issue/PR ever
appears in two splits.

Cross-source overlap caveat: Defects4J bugs reference upstream project
repositories, and GHPR rows come from CNCF projects; these do not overlap
in practice, but if the same upstream issue ever appeared in two sources
its group keys would differ and leakage across *sources* would not be
detected. This limitation is documented in DATASETS.md.

Output: data/final/train.jsonl, validation.jsonl, test.jsonl
plus data/final/split_report.json with the deterministic mapping.

Usage:
    python -m src.data.split
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.utils import (
    PROJECT_ROOT,
    get_group_key,
    load_config,
    read_jsonl,
    write_jsonl,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

PROCESSED_FILES = {
    "aifaultbench": "aifaultbench.jsonl",
    "ghpr": "ghpr.jsonl",
    "defects4j": "defects4j.jsonl",
}


def _assign_groups(
    records: list[dict[str, Any]],
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, list[int]]:
    """Assign record indices to splits group-wise, deterministically."""
    groups: dict[str, list[int]] = defaultdict(list)
    singletons: list[int] = []
    for index, record in enumerate(records):
        group_key = get_group_key(record)
        if group_key:
            groups[f"group:{group_key}"].append(index)
        else:
            singletons.append(index)

    units: list[tuple[str, list[int]]] = [(key, members) for key, members in groups.items()]
    units += [(f"single:{index}", [index]) for index in singletons]

    # Deterministic shuffle: sort by hash first so the input file order
    # cannot leak ordering bias, then use the seeded RNG.
    units.sort(key=lambda item: item[0])
    rng.shuffle(units)

    total = sum(len(members) for _, members in units)
    targets = {
        "train": ratios["train"] * total,
        "validation": ratios["validation"] * total,
        "test": ratios["test"] * total,
    }
    current = {"train": 0, "validation": 0, "test": 0}
    assignment: dict[str, list[int]] = {"train": [], "validation": [], "test": []}

    for unit_name, members in units:
        # Greedy: place the unit where the relative deficit is largest.
        deficits = {
            split: (targets[split] - current[split]) / max(targets[split], 1.0)
            for split in ("train", "validation", "test")
        }
        chosen = max(("train", "validation", "test"), key=lambda s: (deficits[s], s))
        assignment[chosen].extend(members)
        current[chosen] += len(members)

    for split in assignment:
        assignment[split].sort()
    return assignment


def split_files(config: dict, sources: list[str] | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Split the named (or all) processed datasets into final files.

    output_dir defaults to data/final/; tests pass a temporary directory
    to verify determinism without touching the real output.
    """
    split_cfg = config.get("split", {})
    ratios = {
        "train": float(split_cfg.get("train", 0.8)),
        "validation": float(split_cfg.get("validation", 0.1)),
        "test": float(split_cfg.get("test", 0.1)),
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0 (got {ratios})")

    seed = int(config.get("seed", 42))
    rng = random.Random(seed)
    sources = sources or list(PROCESSED_FILES)
    output_dir = output_dir or FINAL_DIR

    all_records: list[dict[str, Any]] = []
    per_source_counts: dict[str, int] = {}
    for source in sources:
        path = PROCESSED_DIR / PROCESSED_FILES[source]
        if not path.is_file():
            print(f"split: {path} not found (convert/clean first?)", file=sys.stderr)
            continue
        records = read_jsonl(path)
        per_source_counts[source] = len(records)
        all_records.extend(records)

    assignment = _assign_groups(all_records, ratios, rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "seed": seed,
        "ratios": ratios,
        "sources": per_source_counts,
        "total": len(all_records),
        "splits": {},
    }

    sizes: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        records = [all_records[i] for i in assignment[split]]
        out_path = output_dir / f"{split}.jsonl"
        sizes[split] = write_jsonl(out_path, records)
        report["splits"][split] = {
            "examples": sizes[split],
            "share": round(sizes[split] / max(len(all_records), 1), 4),
        }

    # Group integrity check: no group key in two splits.
    group_by_split: dict[str, set[str]] = defaultdict(set)
    for split, indices in assignment.items():
        for index in indices:
            key = get_group_key(all_records[index])
            if key:
                group_by_split[split].add(key)
    overlap_keys: set[str] = set()
    split_names = list(group_by_split)
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1:]:
            overlap_keys |= group_by_split[left_name] & group_by_split[right_name]
    report["group_leakage_detected"] = bool(overlap_keys)
    report["groups_per_split"] = {split: len(keys) for split, keys in group_by_split.items()}

    (output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        report = split_files(config, sources=argv or None)
    except ValueError as exc:
        print(f"split: {exc}", file=sys.stderr)
        return 1
    print(f"split: seed={report['seed']} total={report['total']}")
    for split in ("train", "validation", "test"):
        info = report["splits"][split]
        print(f"  {split}: {info['examples']} examples ({info['share']:.1%})")
    print(f"  groups: {report['groups_per_split']}")
    if report["group_leakage_detected"]:
        print("split: WARNING - group leakage detected across splits!", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
