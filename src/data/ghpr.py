"""Convert the GHPR dataset (GitHub issue -> PR relationships) into the
ForgeMind processed schema.

Source: https://github.com/soroushj/ghpr-dataset (CC BY 4.0)
Raw objects: https://github.com/soroushj/ghpr-dataset-raw

Raw layout:
    data/raw/ghpr/dataset/ghpr.csv        # one row per (issue, PR) pair
    data/raw/ghpr/repo_id_map.json        # resolved numeric repo_id -> slug
    data/raw/ghpr/raw/repos/<owner>/<repo>/pull-N.json   (cached objects)

The CSV only records that a merged PR fixed the linked issue (via GitHub
link keywords) — it does NOT contain PR titles/bodies or diffs. The
numeric repo_id gets resolved to owner/name through the GitHub API
(`GET /repositories/{id}`) once per unique id and cached so we don't
re-hit the API on every run. PR descriptions are pulled from the raw
dataset repository via raw.githubusercontent.com when available.

The issue->PR relationship is always reported as an upstream fact ("this
PR was linked to and merged for this issue") — we never claim to have
verified the fix ourselves.

Not every issue/PR pair makes a good training example, so the converter
applies a few quality gates:
  - the issue needs a non-trivial title;
  - pairs with no resolvable PR context are only kept if the issue body
    is substantive on its own, and the assistant content then avoids
    describing any resolution it can't actually see;
  - everything still has to pass the shared cleaning thresholds afterward.

Usage:
    python -m src.data.ghpr
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.data.utils import (
    PROJECT_ROOT,
    build_example,
    get_github_token,
    read_json,
    write_jsonl,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ghpr"
DATASET_DIR = RAW_DIR / "dataset"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "ghpr.jsonl"
REPO_MAP_PATH = RAW_DIR / "repo_id_map.json"

GITHUB_API_ROOT = "https://api.github.com"
RAW_FETCH_ROOT = "https://raw.githubusercontent.com/soroushj/ghpr-dataset-raw/main/repos"

USER_AGENT = "forgemind-data-pipeline"

# The GHPR README documents the dataset covering these 13 CNCF repos
# (collected October 2020). The repo_id values below were cross-checked
# against the GitHub API (`GET /repositories/{id}` -> full_name) and
# against the per-repo instance counts in the GHPR README.
KNOWN_GHPR_REPOS = {
    "106598834": "theupdateframework/specification",
    "11008207": "vitessio/vitess",
    "1918677": "fluent/fluentd",
    "20580498": "kubernetes/kubernetes",
    "43723161": "helm/helm",
    "46089560": "containerd/containerd",
    "48833910": "tikv/tikv",
    "50613991": "goharbor/harbor",
    "54230994": "coredns/coredns",
    "56342508": "jaegertracing/jaeger",
    "65214191": "envoyproxy/envoy",
    "62921553": "rook/rook",
    "6838921": "prometheus/prometheus",
}


def _http_get(url: str, token: str | None = None, accept: str = "application/vnd.github+json") -> bytes | None:
    request = urllib.request.Request(url)
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Accept", accept)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def _load_repo_id_map(unique_ids: list[str], token: str | None) -> dict[str, str]:
    """Resolve numeric repo_id -> owner/name via the GitHub API.

    Falls back to a small static table for the 13 CNCF repos the GHPR
    README documents, so an offline run still gets correct slugs for the
    common case. Live API results win when available and get cached in
    data/raw/ghpr/repo_id_map.json. Anything unresolvable just stays out
    of the map — never guessed.
    """
    cached: dict[str, str] = {}
    if REPO_MAP_PATH.is_file():
        cached = {str(k): str(v) for k, v in (read_json(REPO_MAP_PATH) or {}).items()}

    mapping = dict(cached)
    for repo_id in unique_ids:
        if repo_id in mapping:
            continue
        payload = _http_get(f"{GITHUB_API_ROOT}/repositories/{repo_id}", token=token)
        if payload is not None:
            try:
                data = json.loads(payload.decode("utf-8"))
                full_name = data.get("full_name")
            except (json.JSONDecodeError, UnicodeDecodeError):
                full_name = None
            if full_name:
                mapping[repo_id] = str(full_name)
                time.sleep(0.3)
                continue
        if repo_id in KNOWN_GHPR_REPOS:
            mapping[repo_id] = KNOWN_GHPR_REPOS[repo_id]
            print(f"ghpr: repo_id {repo_id} not resolvable via API, using documented CNCF slug")
        else:
            print(f"ghpr: could not resolve repo_id {repo_id} (continuing without slug)")

    if mapping != cached:
        REPO_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPO_MAP_PATH.write_text(
            json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8"
        )
    return mapping


def _fetch_raw_pr(slug: str, pull_number: str) -> dict[str, Any] | None:
    """Fetch pull-N.json from ghpr-dataset-raw via raw.githubusercontent.com."""
    if not slug or not pull_number:
        return None
    payload = _http_get(
        f"{RAW_FETCH_ROOT}/{slug}/pull-{pull_number}.json",
        accept="application/vnd.github.raw",
    )
    if payload is None:
        return None
    try:
        data = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _ts_to_iso(ts: str | int | None) -> str | None:
    try:
        ts_int = int(ts)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if ts_int <= 0:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_int))


def _clean_text(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned or cleaned.lower() in {"n/a", "na", "none"}:
        return None
    return cleaned


def _clip(text: str, limit: int, suffix: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + suffix


def _format_body(body: str | None, limit: int = 4000) -> str | None:
    body = _clean_text(body)
    if not body:
        return None
    return _clip(body, limit, "\n... [body truncated]")


def _pr_summary_lines(pr_payload: dict[str, Any]) -> list[str]:
    """Build human-readable PR description lines from a raw PR object."""
    lines: list[str] = []
    title = _clean_text(pr_payload.get("title"))
    if title:
        lines.append(f"PR title: {title}")
    body = _format_body(pr_payload.get("body"))
    if body:
        lines.append(f"PR description:\n{body}")
    merged_at = pr_payload.get("merged_at")
    if merged_at:
        lines.append(f"Merged upstream at: {merged_at}")
    stats: list[str] = []
    if pr_payload.get("additions") is not None:
        stats.append(f"+{pr_payload['additions']} additions")
    if pr_payload.get("deletions") is not None:
        stats.append(f"-{pr_payload['deletions']} deletions")
    if pr_payload.get("changed_files") is not None:
        stats.append(f"{pr_payload['changed_files']} files changed")
    if pr_payload.get("commits") is not None:
        stats.append(f"{pr_payload['commits']} commits")
    if stats:
        lines.append("PR size: " + ", ".join(stats))
    return [line for line in lines if line]


def convert_pair(
    row: dict[str, str],
    slug: str | None,
    pr_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert one (issue, PR) row into a ForgeMind example, or None if it doesn't qualify."""
    issue_number = (row.get("issue_number") or "").strip()
    pull_number = (row.get("pull_number") or "").strip()
    issue_title = _clean_text(row.get("issue_title"))
    issue_body = _format_body(row.get("issue_body_md") or row.get("issue_body_plain"))
    merged_at = _ts_to_iso(row.get("pull_merged_at"))
    created_at = _ts_to_iso(row.get("issue_created_at"))

    if not issue_title or not issue_number or not pull_number:
        return None

    pr_lines = _pr_summary_lines(pr_payload) if pr_payload else []
    has_pr_context = bool(pr_lines)
    # quality gate for the no-PR-context case: require a substantive issue body
    has_substantive_body = bool(issue_body) and len(issue_body) >= 80
    if not has_pr_context and not has_substantive_body:
        return None

    user_parts = [
        "A repository contains the following reported issue:",
        "",
        f"Issue #{issue_number}: {issue_title}",
    ]
    if issue_body:
        user_parts += ["", "Issue description:", issue_body]

    if has_pr_context:
        user_parts += [
            "",
            "A pull request was created to resolve it:",
            "",
            *pr_lines,
            "",
            "Analyze the issue and explain what problem it reports and how the "
            "linked pull request addresses it. Base every statement on the "
            "information above; do not speculate beyond it.",
        ]
    else:
        user_parts += [
            "",
            "The dataset records that a merged pull request fixed this issue, but "
            "the pull request details are not available in this snapshot. "
            "Summarize the reported problem precisely and outline what a "
            "maintainer would need to verify before confirming a resolution.",
        ]

    assistant_parts: list[str] = []
    assistant_parts.append("### Issue Summary")
    first_body_line = issue_body.split("\n")[0] if issue_body else ""
    assistant_parts.append(first_body_line or issue_title)

    if has_pr_context:
        pr_title = next((line[len("PR title: "):] for line in pr_lines if line.startswith("PR title: ")), None)
        pr_body = None
        for index, line in enumerate(pr_lines):
            if line.startswith("PR description:\n"):
                pr_body = pr_lines[index][len("PR description:\n"):]
                break
        assistant_parts.append("")
        assistant_parts.append("### Resolution (as recorded upstream)")
        resolution_text = pr_body or pr_title or "(no description recorded)"
        assistant_parts.append(
            f"Pull request #{pull_number} was linked to this issue with a GitHub "
            f"fix keyword and was merged"
            + (f" on {merged_at}" if merged_at else "")
            + f". The change is recorded upstream as: {resolution_text}"
        )
        assistant_parts.append("")
        assistant_parts.append("### Pull Request Footprint")
        stats = next((line for line in pr_lines if line.startswith("PR size: ")), None)
        assistant_parts.append(stats[len("PR size: "):] if stats else "Not recorded in the dataset.")
        note = (
            "The fix association comes from the dataset's keyword linkage; the "
            "actual diff is not part of this dataset, so the code-level fix "
            "cannot be quoted."
        )
        assistant_parts.append("")
        assistant_parts.append(f"Note: {note}")
    else:
        assistant_parts.append("")
        assistant_parts.append("### Resolution")
        assistant_parts.append(
            "Not describable from the available data: the linked pull request's "
            "contents are not in this snapshot. Only the reported problem is "
            "analyzed here; describing a fix would require inventing information."
        )

    assistant_parts.append("")
    assistant_parts.append("### Regression Test Considerations")
    considerations: list[str] = []
    body_sample = (issue_body or "").lower()
    if any(word in body_sample for word in ("crash", "panic", "hang", "deadlock", "leak")):
        considerations.append(
            "The report describes an explicit failure mode; a regression test "
            "should reproduce those conditions and assert normal behavior."
        )
    if any(word in body_sample for word in ("config", "flag", "option", "setting", "parameter")):
        considerations.append(
            "The issue involves configuration or option handling; cover the "
            "affected configuration path with tests."
        )
    if any(word in body_sample for word in ("version", "upgrade", "compatib", "regression")):
        considerations.append(
            "The report references version or compatibility behavior; add a test "
            "that pins the affected version boundary."
        )
    if not considerations:
        considerations.append(
            "Capture the reported conditions in a test that fails before the fix "
            "and passes after it."
        )
    assistant_parts.append("\n".join(f"- {c}" for c in considerations))

    metadata: dict[str, Any] = {
        "source": "ghpr",
        "task": "issue_resolution",
        "project": slug,
        "license": "CC-BY-4.0 (dataset); issue/PR text remains under the upstream repository's terms",
        "issue_number": issue_number,
        "pull_number": pull_number,
        "issue_url": f"https://github.com/{slug}/issues/{issue_number}" if slug else None,
        "pull_url": f"https://github.com/{slug}/pull/{pull_number}" if slug else None,
        "issue_created_at": created_at,
        "pull_merged_at": merged_at,
        # group by the upstream issue, so every example derived from the same
        # issue (across linked PRs) stays together in one split
        "group_key": f"ghpr:{slug or row.get('repo_id', 'unknown')}#{issue_number}",
    }

    return build_example("\n".join(user_parts), "\n".join(assistant_parts), metadata)


def _open_csv(path: Path):
    """Open a GHPR CSV with a raised field limit (some issue bodies run past 128KB)."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    return open(path, "r", encoding="utf-8", newline="")


def _fetch_all_prs(
    needed: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any] | None]:
    """Fetch all needed raw PR objects concurrently.

    raw.githubusercontent.com is a CDN with no meaningful rate limit, so a
    small thread pool is enough to keep this phase down to a couple of
    minutes. The returned dict always covers every requested key —
    failures just map to None.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[tuple[str, str], dict[str, Any] | None] = {}

    def fetch(key: tuple[str, str]) -> tuple[tuple[str, str], dict[str, Any] | None]:
        return key, _fetch_raw_pr(key[0], key[1])

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, key) for key in needed]
        done = 0
        for future in as_completed(futures):
            key, payload = future.result()
            results[key] = payload
            done += 1
            if done % 200 == 0:
                print(f"ghpr: fetched {done}/{len(needed)} raw PR objects", flush=True)
    return results


def convert(config: dict) -> int:
    """Convert the GHPR dataset. Returns a process exit code."""
    csv_path = DATASET_DIR / "ghpr.csv"
    if not csv_path.is_file():
        print("ghpr: ghpr.csv not found; run `python -m src.data.download` first", file=sys.stderr)
        return 1

    settings = config.get("download", {}).get("ghpr", {})
    sample_limit = int(settings.get("sample_limit", 1500))
    fetch_raw = bool(settings.get("fetch_raw_objects", True))
    token = get_github_token()  # optional — GitHub API is just more rate-limited without it

    with _open_csv(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    print(f"ghpr: index contains {len(rows)} issue<->PR pairs")

    # deterministic sampling: stride across repo/pull/issue-sorted rows so
    # the same config always produces the same subset, no randomness involved
    rows.sort(
        key=lambda r: (
            int(r.get("repo_id", 0) or 0),
            int(r.get("pull_number", 0) or 0),
            int(r.get("issue_number", 0) or 0),
        )
    )
    if sample_limit and len(rows) > sample_limit:
        stride = len(rows) / sample_limit
        rows = [rows[int(i * stride)] for i in range(sample_limit)]
    print(f"ghpr: processing {len(rows)} pairs (sample_limit={sample_limit})")

    unique_repo_ids = sorted({(row.get("repo_id") or "").strip() for row in rows} - {""})
    repo_map = _load_repo_id_map(unique_repo_ids, token)
    print(f"ghpr: resolved {len(repo_map)}/{len(unique_repo_ids)} repository ids")

    records: list[dict[str, Any]] = []

    # fetch every unique PR object up front (concurrently, deterministically
    # keyed), then build examples in the original row order
    needed = sorted(
        {
            (repo_map.get((row.get("repo_id") or "").strip()) or "", (row.get("pull_number") or "").strip())
            for row in rows
            if fetch_raw and repo_map.get((row.get("repo_id") or "").strip())
        }
    )
    needed = [(slug, number) for slug, number in needed if slug and number]
    print(f"ghpr: fetching {len(needed)} unique raw PR objects ...", flush=True)
    pr_cache = _fetch_all_prs(needed) if needed else {}
    fetched = sum(1 for payload in pr_cache.values() if payload is not None)

    for row in rows:
        slug = repo_map.get((row.get("repo_id") or "").strip())
        pr_payload = pr_cache.get((slug or "", row.get("pull_number", ""))) if fetch_raw else None
        example = convert_pair(row, slug, pr_payload)
        if example is not None:
            records.append(example)

    count = write_jsonl(OUT_PATH, records)
    print(f"ghpr: {count} examples written -> {OUT_PATH} ({fetched}/{len(needed)} raw PR objects fetched)")
    return 0


if __name__ == "__main__":
    from src.data.utils import load_config

    raise SystemExit(convert(load_config()))