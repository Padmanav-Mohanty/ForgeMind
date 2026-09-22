"""Section the user message of ForgeMind examples into trimmable blocks.

The converters embed user-side evidence as labeled blocks:

    Issue report:
    ```
    <issue body — may itself contain ``` fences>
    ```

so a trim stage can drop or shorten a *whole labeled block* without
corrupting the remainder. Parsing is ANCHOR-based, not naive fence
scanning: a section starts at a line exactly matching a known converter
label (or an instruction opener), and a fenced section ends at the LAST
fence line before the next anchor — so fences nested inside evidence
bodies never break the parse.

Anchors come from the actual converter sources:
- aifaultbench (fenced): "Issue report:", "Minimal reproduction script:",
  "Reproduction output (stdout):", "Reproduction errors (stderr):"
- defects4j (fenced): "Failure output:", "The relevant code under
  suspicion currently reads (removed lines of the fix diff, i.e. the
  buggy version):"
- ghpr (prose, unfenced): "Issue description:", "A pull request was
  created to resolve it:"
- instruction openers (always KEEP): the fixed task paragraphs each
  converter appends

This module is pure text processing — no tokenizer, no I/O — so it is
unit-testable offline. Sections carry EXACT character spans of the source,
so concatenating section texts reproduces the input byte-for-byte
(lossless roundtrip; asserted by tests).
"""

from __future__ import annotations

import re
from typing import Any

# Fenced evidence blocks: label line, then a ``` fence, body, closing fence.
FENCE_LABELS = {
    "Issue report:",
    "Minimal reproduction script:",
    "Reproduction output (stdout):",
    "Reproduction errors (stderr):",
    "Failure output:",
    "The relevant code under suspicion currently reads (removed lines of the fix diff, i.e. the buggy version):",
}

# Prose evidence blocks: label line, then free text until the next anchor.
PROSE_LABELS = {
    "Issue description:",
    "A pull request was created to resolve it:",
}

# Fixed task-instruction paragraphs appended by the converters. A line
# starting with one of these begins the trailing KEEP section.
INSTRUCTION_OPENERS = (
    "Analyze this fault.",
    "Analyze the failure and identify the likely root cause",
    "Identify the failing operation",
    "Analyze the issue and explain what problem it reports",
    "Summarize the reported problem precisely",
    "The dataset records that a merged pull request fixed this issue",
)

_FENCE_RE = re.compile(r"^```")


def _anchor_kind(line: str) -> str | None:
    """Classify a line as a section anchor, or None."""
    stripped = line.rstrip("\n")
    if stripped in FENCE_LABELS:
        return "fence"
    if stripped in PROSE_LABELS:
        return "prose"
    if any(stripped.startswith(opener) for opener in INSTRUCTION_OPENERS):
        return "instruction"
    return None


def section_user_message(text: str) -> list[dict[str, Any]]:
    """Split a user message into ordered, lossless sections.

    Each section is:
        {"kind": "keep"|"evidence", "style": "fence"|"prose"|"none",
         "label": str, "start": int, "end": int, "text": str}
    where [start, end) are character offsets into the original string and
    text == original[start:end]. Evidence sections always carry their
    converter label; keep sections are preamble, titles, and the task
    instruction.
    """
    lines = text.split("\n")
    # line index -> anchor kind
    anchors: dict[int, str] = {}
    for index, line in enumerate(lines):
        kind = _anchor_kind(line)
        if kind:
            anchors[index] = kind

    anchor_lines = sorted(anchors)
    if not anchor_lines:
        return [
            {"kind": "keep", "style": "none", "label": "",
             "start": 0, "end": len(text), "text": text}
        ]

    # character offset of the start of each line
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1

    sections: list[dict[str, Any]] = []
    cursor = 0  # char offset of the not-yet-consumed text
    i = 0
    while i < len(lines):
        if i not in anchors:
            i += 1
            continue
        kind = anchors[i]
        anchor_start = offsets[i]

        if anchor_start > cursor:
            sections.append(
                {"kind": "keep", "style": "none", "label": "",
                 "start": cursor, "end": anchor_start,
                 "text": text[cursor:anchor_start]}
            )
            cursor = anchor_start

        if kind == "instruction":
            sections.append(
                {"kind": "keep", "style": "none", "label": "",
                 "start": cursor, "end": len(text),
                 "text": text[cursor:]}
            )
            cursor = len(text)
            break

        # find the next anchor line (of any kind) after i
        next_anchor_line = next(
            (j for j in anchor_lines if j > i), len(lines)
        )
        if kind == "fence":
            # close fence = last fence line strictly before the next anchor
            close_line = max(
                (j for j in range(i, next_anchor_line) if _FENCE_RE.match(lines[j])),
                default=None,
            )
            if close_line is None:
                # malformed: no fence after the label; treat label+rest as prose
                end_line = next_anchor_line
                style = "prose"
            else:
                end_line = close_line + 1
                style = "fence"
        else:  # prose evidence runs until the next anchor line
            end_line = next_anchor_line
            style = "prose"

        end_offset = offsets[end_line] if end_line < len(lines) else len(text)
        label = lines[i].strip()
        sections.append(
            {"kind": "evidence", "style": style, "label": label,
             "start": cursor, "end": end_offset,
             "text": text[cursor:end_offset]}
        )
        cursor = end_offset
        i = end_line

    if cursor < len(text):
        sections.append(
            {"kind": "keep", "style": "none", "label": "",
             "start": cursor, "end": len(text), "text": text[cursor:]}
        )
    return sections


def section_kinds(sections: list[dict[str, Any]]) -> str:
    """Compact signature like 'keep|evidence|keep' for debugging/tests."""
    return "|".join(section["kind"] for section in sections)


def roundtrip_ok(text: str, sections: list[dict[str, Any]]) -> bool:
    """True when concatenating section texts reproduces the input exactly."""
    return "".join(section["text"] for section in sections) == text


def trim_evidence_section(
    section: dict[str, Any], keep_chars: int
) -> dict[str, Any]:
    """Head-truncate one evidence section's body to ~keep_chars.

    Keeps the section's label and fences intact. The body keeps its HEAD
    (converter excerpts are head-truncated upstream too, and issue
    narratives / stack traces lead with the informative part) and gains a
    deterministic marker stating what was removed, so a trimmed example
    stays honest about its own completeness. Returns the replacement
    section (text only; spans of the ORIGINAL no longer apply).
    """
    if section["kind"] != "evidence":
        raise ValueError("only evidence sections can be trimmed")
    lines = section["text"].split("\n")
    if section["style"] == "fence":
        fence_open = next(
            (k for k, l in enumerate(lines) if _FENCE_RE.match(l)), None
        )
        if fence_open is None:
            raise ValueError("fenced evidence section has no fence")
        close = max(
            (k for k in range(fence_open, len(lines)) if _FENCE_RE.match(lines[k]))
        )
        head = lines[: fence_open + 1]
        tail = [lines[close]]
        body = lines[fence_open + 1 : close]
    else:
        head, tail, body = lines[:1], [], lines[1:]

    body_chars = sum(len(l) + 1 for l in body)
    if body_chars <= keep_chars:
        return {"section": section, "trimmed": False}

    kept: list[str] = []
    used = 0
    for line in body:
        if used + len(line) + 1 > keep_chars:
            break
        kept.append(line)
        used += len(line) + 1
    marker = f"... [evidence trimmed: kept {used} of {body_chars} characters]"
    new_lines = head + kept + [marker] + tail
    new_text = "\n".join(new_lines)
    return {
        "section": {
            "kind": "evidence", "style": section["style"],
            "label": section["label"], "start": section["start"],
            "end": section["end"], "text": new_text,
        },
        "trimmed": True,
        "chars_before": body_chars,
        "chars_after": used,
    }


def drop_evidence_section(section: dict[str, Any]) -> dict[str, Any]:
    """Remove one evidence section (label + block) entirely."""
    if section["kind"] != "evidence":
        raise ValueError("only evidence sections can be dropped")
    return {"section": None, "dropped_label": section["label"]}


def rebuild_user_message(sections: list[dict[str, Any]]) -> str:
    """Rejoin sections after trimming/dropping — LOSSLESS concatenation.

    Section spans partition the source exactly, so plain concatenation of
    surviving section texts preserves every surviving byte (including
    separators carried inside section texts). Deliberately no newline
    collapsing or stripping: untouched content must come through
    byte-identical, cosmetic gaps from dropped sections and all.
    """
    return "".join(s["text"] for s in sections if s is not None)
