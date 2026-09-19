"""Convert AIFaultBench raw data into the ForgeMind processed schema.

Raw layout (downloaded by src.data.download):
    data/raw/aifaultbench/
        index.csv                     # one row per fault, with issue metadata
        bugs/<NNN>/
            manifest.json             # fault identifiers, repo, commit
            bug_report.txt            # TITLE/BODY recovered from GitHub issue
            reproduction.json         # reproducibility evidence + steps
            repro.py                  # minimal reproduction script
            repro_stdout.log          # reproduction output
            repro_stderr.log

Important: AIFaultBench gives us real-world AI software faults *and* the
evidence a reproducer actually observed, but it does NOT hand us a known
fix or a curated root cause. So the converter only builds:
  - a "fault analysis" example per fault (issue + reproduction evidence ->
    structured analysis task), where the assistant content is grounded
    strictly in the dataset's own reproduction evidence; and
  - a "runtime failure diagnosis" example for reproducible faults whose
    logs contain an exception traceback (user supplies the failing
    behavior; assistant describes the observed failure without inventing
    a fix).

Nothing gets fabricated — no root cause, no fix. When the dataset doesn't
have that info, the assistant content just says what's known (observed
failure, evidence, reproduction command) and flags what isn't (confirmed
fix).

Usage:
    python -m src.data.aifaultbench
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any

from src.data.utils import (
    PROJECT_ROOT,
    assistant_message,
    build_example,
    read_json,
    read_text,
    user_message,
    write_jsonl,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "aifaultbench"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "aifaultbench.jsonl"

# Map AIFaultBench domains (as used in index.csv) onto our controlled task
# vocabulary. "Domain" here is the software domain of the fault; the task
# we're actually training is fault analysis / debugging on that domain.
DOMAIN_TASKS = {
    "DL": "debugging",
    "LLM Infra": "debugging",
    "Agentic": "debugging",
    "ML": "debugging",
    "RL": "debugging",
    "AI Tooling": "debugging",
}


def _parse_bug_report(text: str) -> dict[str, str]:
    """Split bug_report.txt into title and body.

    The dataset uses two formats:
      1. ``TITLE\n...\nBODY\n...`` markers (recovered GitHub issue text)
      2. a structured header with ``Title:`` field + ``Issue Body:`` section
    """
    result: dict[str, str] = {"title": "", "body": ""}
    if not text:
        return result
    if re.search(r"^TITLE\s*$", text, re.M):
        # format 1 — may carry a leading "# recovered-from-issue:" comment line
        title_match = re.search(r"^TITLE\s*\n(.*?)(?=^BODY\s*\n|\Z)", text, re.M | re.S)
        body_match = re.search(r"^BODY\s*\n(.*)$", text, re.M | re.S)
        if title_match:
            result["title"] = title_match.group(1).strip()
        if body_match:
            result["body"] = body_match.group(1).strip()
        return result
    # format 2 — structured header lines, body sits after "Issue Body:"
    title_match = re.search(r"^Title:\s*(.+)$", text, re.M)
    if title_match:
        result["title"] = title_match.group(1).strip()
    body_match = re.search(r"^Issue Body:\s*\n?(.*)$", text, re.M | re.S)
    if body_match:
        result["body"] = body_match.group(1).strip()
    if not result["body"]:
        # fall back to everything after the header block (first blank line)
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            result["body"] = parts[1].strip()
    return result


def _extract_exception(log_text: str | None, max_lines: int = 40) -> str | None:
    """Grab the tail (exception + traceback) of a reproduction log."""
    if not log_text:
        return None
    lines = [line.rstrip() for line in log_text.splitlines() if line.strip()]
    if not lines:
        return None
    # walk backwards so we land on the *last* traceback in the log, not the first
    exc_start = None
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if re.search(r"\b([A-Za-z_][\w\.]*\.)*([A-Z]\w*Error|[A-Z]\w*Exception|Warning)\b", line):
            exc_start = idx
    if exc_start is None:
        return None
    keep_from = max(exc_start - 10, 0)
    excerpt = "\n".join(lines[keep_from : exc_start + max_lines])
    return excerpt.strip() or None


def _excerpt(text: str | None, limit: int = 3000) -> str | None:
    if not text:
        return None
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _format_repro_steps(steps: list[Any] | None) -> str | None:
    if not steps:
        return None
    cleaned = [str(step).strip() for step in steps if str(step).strip()]
    if not cleaned:
        return None
    return "\n".join(f"{i}. {step}" for i, step in enumerate(cleaned, start=1))


def _language_for(library: str) -> str:
    # AIFaultBench reproduction packages are all Python-based (repro.py,
    # requirements.txt). The upstream issue itself might be about some
    # other language, but the reproduction evidence we actually have is
    # Python, so that's what we tag.
    return "python"


def _first_error_line(log_text: str | None) -> str | None:
    if not log_text:
        return None
    for line in log_text.splitlines():
        stripped = line.strip()
        if re.search(r"\b([A-Za-z_][\w\.]*\.)*([A-Z]\w*Error|[A-Z]\w*Exception)\b", stripped):
            return stripped[:300]
    return None


def convert_bug(bug_dir: Path, index_row: dict[str, str], config: dict) -> dict[str, Any] | None:
    """Convert a single AIFaultBench bug directory into processed examples."""
    bug_id = bug_dir.name
    manifest = read_json(bug_dir / "manifest.json") or {}
    report = _parse_bug_report(read_text(bug_dir / "bug_report.txt") or "")
    reproduction = read_json(bug_dir / "reproduction.json") or {}
    repro_py = read_text(bug_dir / "repro.py")
    settings = config.get("download", {}).get("aifaultbench", {})
    max_log_bytes = int(settings.get("max_log_bytes", 20000))

    stdout_log = read_text(bug_dir / "repro_stdout.log", max_bytes=max_log_bytes)
    stderr_log = read_text(bug_dir / "repro_stderr.log", max_bytes=max_log_bytes)

    issue_url = index_row.get("issue_url") or manifest.get("issue_url") or ""
    repository = index_row.get("repository") or ""
    domain = index_row.get("domain") or ""
    labels = index_row.get("labels") or ""
    reproducible_raw = reproduction.get("reproducible", index_row.get("reproducible", ""))
    reproducible = bool(reproducible_raw) if str(reproducible_raw).lower() not in {"", "none"} else None

    if not report["body"] and not report["title"]:
        return None  # nothing worth building an example out of

    title_line = f"Issue title: {report['title']}" if report["title"] else "Issue title: (unavailable)"

    # Example 1 — fault analysis (root-cause style task, evidence-grounded)

    user_parts = [
        "The following is a reported fault in an open-source AI software project.",
        "",
        title_line,
        "",
        "Issue report:",
        "```",
        report["body"] or "(empty issue body)",
        "```",
    ]

    repro_steps = _format_repro_steps(reproduction.get("steps"))
    evidence = reproduction.get("evidence")
    if reproducible is True and (evidence or repro_steps):
        user_parts.append("")
        user_parts.append("Reproduction attempt:")
        if repro_steps:
            user_parts.append(repro_steps)
        if evidence:
            user_parts.append("")
            user_parts.append(f"Observed evidence: {evidence}")
    elif reproducible is False and reproduction.get("blocking_reason"):
        user_parts.append("")
        user_parts.append(
            f"Reproduction attempt: not reproducible. Reported blocking reason: "
            f"{reproduction.get('blocking_reason')}"
        )

    if repro_py and repro_py.strip():
        user_parts.append("")
        user_parts.append("Minimal reproduction script:")
        user_parts.append("```python")
        user_parts.append(_excerpt(repro_py, 2500) or "")
        user_parts.append("```")

    if stdout_log and stdout_log.strip():
        excerpt = _excerpt(stdout_log, 2000)
        if excerpt:
            user_parts.append("")
            user_parts.append("Reproduction output (stdout):")
            user_parts.append("```")
            user_parts.append(excerpt)
            user_parts.append("```")

    if stderr_log and stderr_log.strip():
        excerpt = _excerpt(stderr_log, 2000)
        if excerpt:
            user_parts.append("")
            user_parts.append("Reproduction errors (stderr):")
            user_parts.append("```")
            user_parts.append(excerpt)
            user_parts.append("```")

    user_parts.append("")
    user_parts.append(
        "Analyze this fault. Describe what the evidence shows about the failure, "
        "assess what category of problem it is, and recommend how a maintainer "
        "should proceed (investigation steps and what tests would guard against "
        "regressions). Only state conclusions the evidence supports; clearly "
        "mark anything uncertain."
    )

    assistant_parts: list[str] = []

    # observed-failure summary — strictly from the issue + reproduction evidence,
    # nothing inferred beyond what's actually in the logs
    assistant_parts.append("### Observed Failure")
    failure_summary = _first_error_line(stdout_log) or _first_error_line(stderr_log)
    if failure_summary:
        assistant_parts.append(
            f"The reproduction ends with: `{failure_summary}` The issue report "
            f"describes this behavior as the fault under investigation."
        )
    elif evidence:
        assistant_parts.append(str(evidence))
    else:
        assistant_parts.append(
            "The reported behavior is the issue body above; no reproduction "
            "output is available in the dataset for this fault."
        )

    assistant_parts.append("")
    assistant_parts.append("### Evidence")
    evidence_lines: list[str] = []
    if report["title"]:
        evidence_lines.append(f"- Issue: {report['title']}")
    if repository:
        evidence_lines.append(f"- Affected repository: {repository}")
    if domain:
        evidence_lines.append(f"- Fault domain (AIFaultBench classification): {domain}")
    if labels:
        evidence_lines.append(f"- Issue labels: {labels.replace(';', ', ')}")
    if reproducible is not None:
        state = "verified reproducible" if reproducible else "not reproducible in the benchmark environment"
        evidence_lines.append(f"- Reproduction status: {state}")
    if reproduction.get("reproduction_command"):
        evidence_lines.append(f"- Reproduction command: `{reproduction['reproduction_command']}`")
    assistant_parts.append("\n".join(evidence_lines) if evidence_lines else "- (no additional evidence fields available)")

    # category hypothesis — only allowed as an explicitly hedged inference,
    # never stated as fact
    assistant_parts.append("")
    assistant_parts.append("### Assessment")
    hypothesis_bits: list[str] = []
    if failure_summary:
        hypothesis_bits.append("the reproduction terminates in a raised exception, so this presents as a runtime failure")
    elif evidence:
        hypothesis_bits.append("the reproduction evidence confirms the reported misbehavior")
    if labels:
        label_list = [l.strip().lower() for l in labels.split(";") if l.strip()]
        if "type:bug" in label_list or "bug" in label_list:
            hypothesis_bits.append('the issue is labeled "bug" upstream')
    if hypothesis_bits:
        assistant_parts.append(
            "Based on the evidence: " + "; ".join(hypothesis_bits) + ". "
            "The dataset does not record a confirmed root cause or fix for this "
            "fault, so any deeper diagnosis would require inspection of the "
            "repository at the recorded commit."
        )
    else:
        assistant_parts.append(
            "The dataset does not record a confirmed root cause or fix for this "
            "fault. A diagnosis would require inspection of the repository at "
            "the recorded commit."
        )

    assistant_parts.append("")
    assistant_parts.append("### Recommended Next Steps")
    next_steps: list[str] = []
    if manifest.get("commit_hash"):
        next_steps.append(
            f"Check out {repository or 'the repository'} at commit "
            f"`{manifest['commit_hash']}` to inspect the code that exhibits the fault."
        )
    next_steps.append("Re-run the reproduction script in an isolated environment to confirm the failure locally.")
    if reproducible is True:
        next_steps.append(
            "Bisect the failure against the reproduction script to isolate the change "
            "that introduced the behavior."
        )
    next_steps.append("Review the issue discussion for maintainer-provided context before proposing a fix.")
    assistant_parts.append("\n".join(f"{i}. {step}" for i, step in enumerate(next_steps, start=1)))

    assistant_parts.append("")
    assistant_parts.append("### Regression Tests")
    if reproducible is True:
        assistant_parts.append(
            "- Promote the reproduction script into an automated regression test that "
            "asserts the failure mode is gone (e.g., asserting the previously faulty "
            "output or exception no longer occurs).\n"
            "- Add a test at the boundary where the failure first appears, based on the "
            "reproduction inputs above.\n"
            "- Add an environment/dependency smoke test covering the versions recorded "
            "in the reproduction requirements."
        )
    else:
        assistant_parts.append(
            "- Add a regression test once the failure is reproduced; the dataset does "
            "not include a passing reproduction for this fault."
        )

    metadata_analysis: dict[str, Any] = {
        "source": "aifaultbench",
        "task": DOMAIN_TASKS.get(domain, "debugging"),
        "language": _language_for(str(index_row.get("library", ""))),
        "project": repository or None,
        "license": "CC-BY-4.0 (benchmark); upstream issue/repo content remains under its own license",
        "bug_id": bug_id,
        "domain": domain or None,
        "issue_url": issue_url or None,
        "reproducible": reproducible,
        "group_key": f"aifaultbench:{bug_id}",
    }

    examples: list[dict[str, Any]] = [
        build_example("\n".join(user_parts), "\n".join(assistant_parts), metadata_analysis)
    ]

    # Example 2 — runtime failure diagnosis straight from the traceback
    # (only when the logs actually contain an exception; stays grounded
    # in the log text, no extra reasoning bolted on)
    settings = config.get("download", {}).get("aifaultbench", {})
    if settings.get("fetch_logs", True):
        exception_excerpt = _extract_exception(stdout_log) or _extract_exception(stderr_log)
        if exception_excerpt and reproducible is True:
            diag_user = "\n".join(
                [
                    "A Python program from an open-source AI project fails with the following error output:",
                    "",
                    "```",
                    exception_excerpt,
                    "```",
                    "",
                    "Identify the failing operation and explain what the error indicates. "
                    "State only what the output supports.",
                ]
            )
            first_line = exception_excerpt.splitlines()[-1]
            diag_assistant = "\n".join(
                [
                    "### Error Analysis",
                    f"The process terminates with: `{first_line}`",
                    "",
                    "### What the Traceback Shows",
                    "The stack trace above pinpoints the failing call chain; the deepest "
                    "application frame is where the error originates, and the frames above "
                    "it show how execution reached that point.",
                    "",
                    "### What This Does Not Establish",
                    "This output alone does not identify the commit, maintainer fix, or a "
                    "confirmed root cause beyond the failing operation itself. The source "
                    "issue for context is: " + (issue_url or "(not recorded)") + ".",
                ]
            )
            metadata_diag = dict(metadata_analysis)
            metadata_diag["task"] = "error_analysis"
            examples.append(build_example(diag_user, diag_assistant, metadata_diag))

    return {"examples": examples}


def convert(config: dict) -> int:
    """Convert the full AIFaultBench raw dataset. Returns 0 on success."""
    if not RAW_DIR.is_dir():
        print("aifaultbench: raw data not found; run `python -m src.data.download` first", file=sys.stderr)
        return 1

    index_path = RAW_DIR / "index.csv"
    if not index_path.is_file():
        print("aifaultbench: index.csv missing from raw download", file=sys.stderr)
        return 1

    rows: dict[str, dict[str, str]] = {}
    with open(index_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            bug_id = str(row.get("bug_id", "")).strip()
            if bug_id:
                rows[bug_id.zfill(3)] = row

    bug_root = RAW_DIR / "bugs"
    converted = 0
    skipped = 0
    records: list[dict[str, Any]] = []

    for bug_dir in sorted(bug_root.iterdir()):
        if not bug_dir.is_dir():
            continue
        row = rows.get(bug_dir.name, {})
        result = convert_bug(bug_dir, row, config)
        if result is None:
            skipped += 1
            continue
        records.extend(result["examples"])
        converted += 1

    count = write_jsonl(OUT_PATH, records)
    print(f"aifaultbench: {converted} faults converted ({skipped} skipped), {count} examples -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    from src.data.utils import load_config

    raise SystemExit(convert(load_config()))