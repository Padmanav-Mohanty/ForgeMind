"""Download the three ForgeMind source datasets into data/raw/.

Usage:
    python -m src.data.download                  # all datasets
    python -m src.data.download --dataset aifaultbench
    python -m src.data.download --dataset aifaultbench --light   # index only
    python -m src.data.download --skip aifaultbench

Raw data is written under:
    data/raw/aifaultbench/   (Hugging Face dataset mehilshah/AIFaultBench)
    data/raw/ghpr/           (GitHub: soroushj/ghpr-dataset, + selective
                              raw objects from soroushj/ghpr-dataset-raw)
    data/raw/defects4j/      (GitHub: rjust/defects4j, sparse checkout of
                              framework metadata and patches only)

Raw data is never modified after download; re-running is idempotent and
skips directories that already exist. Delete a directory to re-download.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from src.data.utils import PROJECT_ROOT, get_hf_token, load_config

RAW_DIR = PROJECT_ROOT / "data" / "raw"

GHPR_DATASET_URL = "https://github.com/soroushj/ghpr-dataset"
DEFECTS4J_URL = "https://github.com/rjust/defects4j"
AIFAULTBENCH_REPO = "mehilshah/AIFaultBench"

# Always-downloaded AIFaultBench files (metadata, licensing, citation).
AIFAULTBENCH_CORE_PATTERNS = [
    "index.csv",
    "README.md",
    "LICENSE",
    "CITATION.cff",
]

# Per-bug files needed by the converter.
AIFAULTBENCH_BUG_PATTERNS = [
    "bugs/*/manifest.json",
    "bugs/*/bug_report.txt",
    "bugs/*/reproduction.json",
    "bugs/*/repro.py",
]

AIFAULTBENCH_LOG_PATTERNS = [
    "bugs/*/repro_stdout.log",
    "bugs/*/repro_stderr.log",
]

# Defects4J sparse-checkout patterns (gitignore-style, last match wins):
# root files (license.txt, README.md) plus the three metadata groups we
# need. Project source trees, build files, and bundled jars are excluded
# on purpose: they are large and upstream-licensed.
DEFECTS4J_SPARSE_PATTERNS = [
    "/*",
    "!/*/",
    "framework/projects/*/active-bugs.csv",
    "framework/projects/*/patches/**",
    "framework/projects/*/trigger_tests/**",
]


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )


def download_aifaultbench(config: dict, light: bool = False) -> Path:
    """Download the AIFaultBench dataset from Hugging Face.

    Resumable: interrupted downloads are detected via a completion marker
    and continued on the next run (snapshot_download skips files that are
    already complete).
    """
    from huggingface_hub import snapshot_download

    settings = config.get("download", {}).get("aifaultbench", {})
    fetch_logs = bool(settings.get("fetch_logs", True))
    dest = RAW_DIR / "aifaultbench"
    marker = dest / ".download_complete"

    if marker.is_file() and (not light or (dest / "index.csv").is_file()):
        print(f"aifaultbench: already complete at {dest} (delete {marker.name} to re-download)")
        return dest

    if light:
        patterns = list(AIFAULTBENCH_CORE_PATTERNS)
    else:
        patterns = AIFAULTBENCH_CORE_PATTERNS + AIFAULTBENCH_BUG_PATTERNS
        if fetch_logs:
            patterns += AIFAULTBENCH_LOG_PATTERNS

    print(f"aifaultbench: downloading {len(patterns)} file patterns from {AIFAULTBENCH_REPO} ...")
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=AIFAULTBENCH_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns,
        token=get_hf_token(),
        max_workers=8,
    )
    marker.write_text("ok\n", encoding="utf-8")
    print(f"aifaultbench: saved to {dest}")
    return dest


def _clone_repo(url: str, dest: Path) -> None:
    if dest.is_dir() and any(dest.iterdir()):
        print(f"{dest.name}: already present at {dest} (delete to re-download)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {url} -> {dest} ...")
    _run_git(["clone", "--depth", "1", url, str(dest)])
    print(f"cloned into {dest}")


def download_ghpr(config: dict) -> Path:
    """Clone the GHPR dataset repository (index CSV + license)."""
    dest = RAW_DIR / "ghpr" / "dataset"
    _clone_repo(GHPR_DATASET_URL, dest)
    return dest


def download_defects4j(config: dict) -> Path:
    """Sparse-clone the Defects4J framework metadata and patches."""
    dest = RAW_DIR / "defects4j"
    marker = dest / ".download_complete"
    if marker.is_file():
        print(f"defects4j: already complete at {dest} (delete {marker.name} to re-download)")
        return dest
    if dest.is_dir() and any(dest.iterdir()):
        # Partial checkout from an interrupted run: remove and retry.
        print("defects4j: partial checkout detected, re-cloning ...")
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {DEFECTS4J_URL} (sparse) -> {dest} ...")
    _run_git(
        [
            "clone",
            "--depth", "1",
            "--filter=blob:none",
            "--no-checkout",
            DEFECTS4J_URL,
            str(dest),
        ]
    )
    _run_git(["sparse-checkout", "init", "--no-cone"], cwd=dest)
    _run_git(["sparse-checkout", "set", *DEFECTS4J_SPARSE_PATTERNS], cwd=dest)
    _run_git(["checkout", "master"], cwd=dest)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"defects4j: sparse checkout saved to {dest}")
    return dest


DATASET_DOWNLOADERS = {
    "aifaultbench": download_aifaultbench,
    "ghpr": download_ghpr,
    "defects4j": download_defects4j,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ForgeMind raw datasets.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DOWNLOADERS),
        action="append",
        help="dataset to download (repeatable; default: all enabled in config)",
    )
    parser.add_argument("--skip", action="append", default=[], help="dataset to skip")
    parser.add_argument(
        "--light",
        action="store_true",
        help="aifaultbench only: download the index metadata without per-bug files",
    )
    parser.add_argument("--config", default=None, help="path to data_config.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config) if args.config else None)
    enabled = config.get("datasets", {})

    if args.dataset:
        targets = args.dataset
    else:
        targets = [name for name, on in enabled.items() if on]

    targets = [t for t in targets if t not in args.skip]

    failures = 0
    for name in targets:
        if name not in DATASET_DOWNLOADERS:
            print(f"unknown dataset: {name}", file=sys.stderr)
            failures += 1
            continue
        if not enabled.get(name, False):
            print(f"{name}: disabled in config, skipping")
            continue
        try:
            if name == "aifaultbench":
                download_aifaultbench(config, light=args.light)
            else:
                DATASET_DOWNLOADERS[name](config)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{name}: download FAILED: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
