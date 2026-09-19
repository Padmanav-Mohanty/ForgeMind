"""Convert Defects4J benchmark metadata into the ForgeMind processed schema.

Source: https://github.com/rjust/defects4j
Framework license: MIT (license.txt in the repository).
Upstream project source code: each benchmarked project keeps its own
license (Apache-2.0, BSD, LGPL, ...). Because of that, this pipeline only
downloads framework metadata (bug tables, trigger-test outputs, and the
developer patches shipped inside the framework) and never bundles or
re-distributes the actual project source trees. Patches quote small
excerpts of the buggy code needed to describe the fix; DATASETS.md
documents the implications and the per-project license responsibility.

Raw layout (sparse checkout by src.data.download):
    data/raw/defects4j/
        license.txt
        framework/projects/<Project>/active-bugs.csv
            columns: bug.id, revision.id.buggy, revision.id.fixed,
                     report.id, report.url
        framework/projects/<Project>/patches/<bug>.src.patch
        framework/projects/<Project>/patches/<bug>.test.patch   (optional)
        framework/projects/<Project>/trigger_tests/<bug>

Example design (everything below is derived strictly from those files):
  user: failing trigger-test name + stack trace + the relevant buggy-code
        hunk (the '-' lines of the src patch, i.e. the buggy state)
  assistant: root-cause sketch (from the diff), bug location (files
        touched by the patch), the developer fix as a patch, and the
        regression test patch. No invented narrative — if the metadata
        doesn't say it, it doesn't go in.

Usage:
    python -m src.data.defects4j
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any

from src.data.utils import (
    PROJECT_ROOT,
    build_example,
    read_text,
    write_jsonl,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "defects4j"
PROJECTS_DIR = RAW_DIR / "framework" / "projects"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "defects4j.jsonl"

MAX_PATCH_CHARS = 6000
MAX_TRIGGER_CHARS = 3000


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _parse_patch_files(patch_text: str) -> list[dict[str, str]]:
    """Split a unified diff into per-file chunks and strip the diff headers."""
    chunks: list[dict[str, str]] = []
    current_files: dict[str, str] = {}
    body_lines: list[str] = []

    def flush() -> None:
        if current_files.get("file") and body_lines:
            chunks.append({"file": current_files["file"], "body": "\n".join(body_lines).strip()})
        body_lines.clear()

    for line in patch_text.splitlines():
        if line.startswith("diff ") or line.startswith("Index: "):
            # the --- / +++ headers right after this carry the real paths,
            # so just close out the previous chunk here
            flush()
        elif line.startswith("--- "):
            flush()
            path = line[4:].strip()
            current_files = {"file": path[2:] if path.startswith("a/") else path}
        elif line.startswith("+++ "):
            path = line[4:].strip()
            if current_files.get("file") in {"", "/dev/null"}:
                current_files["file"] = path[2:] if path.startswith("b/") else path
        else:
            body_lines.append(line)
    flush()
    return chunks


def _hunk_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for chunk in _parse_patch_files(patch_text):
        if chunk["file"] not in files and chunk["file"] not in {"", "/dev/null"}:
            files.append(chunk["file"])
    return files


def _buggy_hunk(patch_text: str, max_chars: int = 2500) -> str | None:
    """Pull out the removed ('-') lines of a src patch — that's the buggy code."""
    hunks: list[str] = []
    current: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("@@"):
            if current:
                hunks.append("\n".join(current))
                current = []
            current.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            current.append(line)
        elif line.startswith(" ") or line.startswith("+"):
            if current and current[-1].startswith("@@") is False:
                pass
    if current:
        hunks.append("\n".join(current))
    if not hunks:
        return None
    text = "\n\n".join(hunks).strip()
    if not text:
        return None
    return _clip(text, max_chars)


def _parse_trigger_test(text: str) -> dict[str, str]:
    """Parse a trigger_tests/<bug> file — a JUnit test name + exception trace."""
    lines = text.splitlines()
    test_name = ""
    trace_start = 0
    for index, line in enumerate(lines):
        if line.startswith("--- "):
            test_name = line[4:].strip()
            trace_start = index + 1
            break
    trace = "\n".join(lines[trace_start:]).strip()
    # cut off the Ant/JUnit runner frames at the bottom, they're just noise
    runner_markers = ("at org.apache.tools.ant", "at org.junit.runner", "at junit.framework", "at java.lang.reflect")
    cleaned: list[str] = []
    for line in trace.splitlines():
        if any(marker in line for marker in runner_markers):
            break
        cleaned.append(line)
    return {"test": test_name, "trace": "\n".join(cleaned).strip()}


def _first_stack_frames(trace: str, count: int = 3) -> str:
    frames = [line.strip() for line in trace.splitlines() if line.strip().startswith("at ")]
    return "\n".join(frames[:count])


def convert_bug(project: str, bug_row: dict[str, str], config: dict) -> dict[str, Any] | None:
    """Convert one Defects4J bug into a ForgeMind example, or None if there's nothing usable."""
    bug_id = (bug_row.get("bug.id") or "").strip()
    if not bug_id:
        return None
    report_id = (bug_row.get("report.id") or "").strip()
    report_url = (bug_row.get("report.url") or "").strip()
    revision_buggy = (bug_row.get("revision.id.buggy") or "").strip()
    revision_fixed = (bug_row.get("revision.id.fixed") or "").strip()

    patches_dir = PROJECTS_DIR / project / "patches"
    src_patch = read_text(patches_dir / f"{bug_id}.src.patch") or ""
    include_test_patches = config.get("download", {}).get("defects4j", {}).get("include_test_patches", True)
    test_patch = read_text(patches_dir / f"{bug_id}.test.patch") or "" if include_test_patches else ""
    trigger_text = read_text(PROJECTS_DIR / project / "trigger_tests" / bug_id) or ""

    if not src_patch.strip() and not trigger_text.strip():
        return None

    trigger = _parse_trigger_test(trigger_text) if trigger_text else {"test": "", "trace": ""}
    src_files = _hunk_files(src_patch)
    test_files = _hunk_files(test_patch)

    # user content — the failing behavior plus the buggy code context
    user_parts: list[str] = [
        f"The following test fails in the Java project '{project}' at revision "
        f"{revision_buggy or '(unknown)'}:",
        "",
    ]
    if trigger["test"]:
        user_parts.append(f"Failing test: {trigger['test']}")
        user_parts.append("")
    if trigger["trace"]:
        user_parts.append("Failure output:")
        user_parts.append("```")
        user_parts.append(_clip(trigger["trace"], MAX_TRIGGER_CHARS))
        user_parts.append("```")
        user_parts.append("")
    buggy_code = _buggy_hunk(src_patch)
    if buggy_code:
        user_parts.append("The relevant code under suspicion currently reads (removed lines of the fix diff, i.e. the buggy version):")
        user_parts.append("```diff")
        user_parts.append(_clip(buggy_code, 2500))
        user_parts.append("```")
        user_parts.append("")
    user_parts.append(
        "Analyze the failure and identify the likely root cause, where the bug "
        "is located, and how it should be fixed. Ground every statement in the "
        "information above."
    )

    # assistant content — root cause + location + fix + regression test,
    # all pulled from the developer patch (that's D4J's ground truth)
    assistant_parts: list[str] = []
    assistant_parts.append("### Root Cause")
    if buggy_code:
        removed_lines = [
            line[1:].strip() for line in buggy_code.splitlines()
            if line.startswith("-") and not line.startswith("---")
        ]
        removed_lines = [line for line in removed_lines if line][:6]
        if removed_lines:
            assistant_parts.append(
                "The developer fix removes the following faulty logic, which is "
                "therefore the defective code responsible for the failing test:"
            )
            assistant_parts.append("```java")
            assistant_parts.extend(removed_lines)
            assistant_parts.append("```")
        else:
            assistant_parts.append(
                "The fix modifies the code shown above; the exact defect is the "
                "difference between the buggy hunk and the fixed version."
            )
    else:
        assistant_parts.append(
            "The failing test output above is the recorded defect trigger; the "
            "patch does not expose removable lines, so the defect is best "
            "located via the stack trace's deepest project frame."
        )

    assistant_parts.append("")
    assistant_parts.append("### Bug Location")
    if src_files:
        assistant_parts.extend(f"- `{path}`" for path in src_files)
    else:
        assistant_parts.append("- (patch does not record file paths)")

    assistant_parts.append("")
    assistant_parts.append("### Fix (developer patch)")
    if src_patch.strip():
        assistant_parts.append("```diff")
        assistant_parts.append(_clip(src_patch.strip(), MAX_PATCH_CHARS))
        assistant_parts.append("```")
    else:
        assistant_parts.append("No source patch is recorded for this bug in the framework metadata.")

    assistant_parts.append("")
    assistant_parts.append("### Regression Test")
    if test_patch.strip():
        assistant_parts.append(
            "The benchmark records the following regression test added with the fix:"
        )
        assistant_parts.append("```diff")
        assistant_parts.append(_clip(test_patch.strip(), MAX_PATCH_CHARS))
        assistant_parts.append("```")
    else:
        assistant_parts.append(
            "No dedicated test patch is recorded for this bug; the trigger test "
            + (f"({trigger['test']}) " if trigger["test"] else "")
            + "serves as the regression signal."
        )

    metadata: dict[str, Any] = {
        "source": "defects4j",
        "task": "bug_fix",
        "language": "java",
        "project": project,
        "license": (
            "Defects4J framework metadata is MIT; the underlying project "
            f"({project}) and its source remain under their own upstream license"
        ),
        "bug_id": f"{project}-{bug_id}",
        "report_id": report_id or None,
        "report_url": report_url or None,
        "revision_buggy": revision_buggy or None,
        "revision_fixed": revision_fixed or None,
        "trigger_test": trigger["test"] or None,
        "group_key": f"defects4j:{project}#{bug_id}",
    }

    return build_example("\n".join(user_parts), "\n".join(assistant_parts), metadata)


def convert(config: dict) -> int:
    """Convert all Defects4J project metadata. Returns a process exit code."""
    if not PROJECTS_DIR.is_dir():
        print("defects4j: raw data not found; run `python -m src.data.download` first", file=sys.stderr)
        return 1

    license_text = read_text(RAW_DIR / "license.txt") or ""
    framework_mit = "Permission is hereby granted" in license_text
    if not framework_mit:
        print(
            "defects4j: WARNING - could not verify the framework MIT license "
            "text in the raw checkout; metadata license strings still note "
            "the framework origin.",
            file=sys.stderr,
        )

    records: list[dict[str, Any]] = []
    total_bugs = 0
    converted_bugs = 0

    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        bugs_csv = project_dir / "active-bugs.csv"
        if not bugs_csv.is_file():
            continue
        project = project_dir.name
        with open(bugs_csv, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        total_bugs += len(rows)
        for row in rows:
            example = convert_bug(project, row, config)
            if example is not None:
                records.append(example)
                converted_bugs += 1

    count = write_jsonl(OUT_PATH, records)
    print(
        f"defects4j: {converted_bugs}/{total_bugs} bugs converted, "
        f"{count} examples -> {OUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    from src.data.utils import load_config

    raise SystemExit(convert(load_config()))