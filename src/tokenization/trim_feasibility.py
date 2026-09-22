"""Read-only feasibility analysis: can over-cap examples be rescued?

For every example whose chat-template token length exceeds a candidate
max_seq_length, this module simulates — WITHOUT modifying any file — a
deterministic user-side trimming strategy:

  1. the user message is sectioned (src/tokenization/user_sections.py);
  2. evidence sections are head-truncated to a character budget, in a
     fixed priority order (noisiest evidence first);
  3. if still over the cap, the least informative evidence sections are
     DROPPED entirely, again in a fixed order;
  4. the trimmed user message is re-measured with the same chat-template
     token counting as the training path (src/tokenization/render.py).

Nothing is written back: the simulator returns measurements only. The
outcome classification is deliberately conservative:

  rescue_full      — fits after user-side trimming only (completion intact)
  rescue_partial   — fits only after ALSO capping the completion, i.e. the
                     training target itself would have to be altered
                     (reported for review, NOT counted as safe)
  not_rescuable    — even dropping every trimmable evidence section cannot
                     get under the cap (the fixed instruction + preamble +
                     completion alone exceed it)

Quality guards recorded per example:
  - evidence retained as a share of the original evidence characters;
  - whether any evidence section was dropped outright;
  - hard floors: examples whose instruction block itself exceeds the cap,
    or that need completion alteration, are never reported as safe.

Usage (module entry point):
    python -m src.tokenization.trim_feasibility
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.utils import PROJECT_ROOT, get_user_content
from src.tokenization.render import (
    completion_token_span,
    count_tokens,
    extract_conversation,
)
from src.tokenization.user_sections import (
    section_user_message,
    trim_evidence_section,
)

FINAL_DIR = PROJECT_ROOT / "data" / "final"
SPLIT_FILES = ("train", "validation", "test")

# Trim priority within the user message: noisiest / least information-dense
# evidence first. Derived from the converters: logs repeat environment
# chatter, the repro script is boilerplate around a few key lines, the
# buggy-code hunk is dense, the issue body is the core narrative.
TRIM_PRIORITY = (
    "Reproduction output (stdout):",
    "Reproduction errors (stderr):",
    "Minimal reproduction script:",
    "Failure output:",
    "Issue report:",
    "The relevant code under suspicion currently reads (removed lines of the fix diff, i.e. the buggy version):",
)

# Character budget per evidence section when trimming (deterministic).
SECTION_CHAR_BUDGET = 1200


@dataclass
class TrimPlan:
    """Result of simulating the deterministic trim for one example."""

    index: int
    split: str
    source: str
    task: str
    original_tokens: int
    completion_tokens: int
    fits_after_user_trim: bool
    final_tokens: int
    evidence_chars_before: int
    evidence_chars_after: int
    trimmed_sections: list[str] = field(default_factory=list)
    dropped_sections: list[str] = field(default_factory=list)
    completion_would_need_trimming: bool = False
    completion_only_tokens: int = 0  # tokens if user evidence were minimal
    notes: list[str] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        if self.fits_after_user_trim:
            return "rescue_full"
        if self.completion_would_need_trimming:
            return "rescue_partial"
        return "not_rescuable"

    @property
    def evidence_retained_share(self) -> float:
        if self.evidence_chars_before == 0:
            return 1.0
        return self.evidence_chars_after / self.evidence_chars_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "split": self.split,
            "source": self.source,
            "task": self.task,
            "outcome": self.outcome,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "completion_tokens": self.completion_tokens,
            "evidence_chars_before": self.evidence_chars_before,
            "evidence_chars_after": self.evidence_chars_after,
            "evidence_retained_share": round(self.evidence_retained_share, 3),
            "trimmed_sections": self.trimmed_sections,
            "dropped_sections": self.dropped_sections,
            "completion_would_need_trimming": self.completion_would_need_trimming,
            "notes": self.notes,
        }


def _evidence_sections(user_text: str) -> list[dict[str, Any]]:
    sections = section_user_message(user_text)
    return [s for s in sections if s["kind"] == "evidence"]


def simulate_trim(
    example: dict[str, Any],
    tokenizer: Any,
    cap: int,
    section_char_budget: int = SECTION_CHAR_BUDGET,
) -> TrimPlan:
    """Simulate the deterministic trim for one example. Read-only."""
    conversation = extract_conversation(example)
    if conversation is None:
        raise ValueError("example has no renderable conversation")
    user_text = get_user_content(example)
    evidence = _evidence_sections(user_text)
    evidence_before = sum(len(s["text"]) for s in evidence)

    plan = TrimPlan(
        index=-1,
        split="",
        source=(example.get("metadata") or {}).get("source", ""),
        task=(example.get("metadata") or {}).get("task", ""),
        original_tokens=count_tokens(conversation, tokenizer),
        completion_tokens=completion_token_span(conversation, tokenizer),
        fits_after_user_trim=False,
        final_tokens=0,
        evidence_chars_before=evidence_before,
        evidence_chars_after=evidence_before,
    )

    trimmed_sections = section_user_message(user_text)

    def measure(sections: list[dict[str, Any]]) -> int:
        rebuilt = "".join(s["text"] for s in sections if s is not None)
        candidate = [dict(conversation[0], content=rebuilt)] + conversation[1:]
        return count_tokens(candidate, tokenizer)

    current = plan.original_tokens
    if current <= cap:
        plan.fits_after_user_trim = True
        plan.final_tokens = current
        return plan

    # pass 1: head-truncate evidence sections, noisiest priority first
    ordered = sorted(
        range(len(trimmed_sections)),
        key=lambda i: (
            TRIM_PRIORITY.index(trimmed_sections[i]["label"])
            if trimmed_sections[i]["label"] in TRIM_PRIORITY
            else len(TRIM_PRIORITY),
            i,
        ),
    )
    for section_index in ordered:
        if current <= cap:
            break
        section = trimmed_sections[section_index]
        if section["kind"] != "evidence":
            continue
        result = trim_evidence_section(section, section_char_budget)
        if result["trimmed"]:
            trimmed_sections[section_index] = result["section"]
            plan.trimmed_sections.append(section["label"])
            current = measure(trimmed_sections)

    # pass 2: drop remaining over-budget evidence sections entirely,
    # least-informative first, until under cap or nothing left to drop
    if current > cap:
        droppable = [
            i
            for i in ordered
            if trimmed_sections[i]["kind"] == "evidence"
        ]
        for section_index in droppable:
            if current <= cap:
                break
            dropped = trimmed_sections[section_index]
            trimmed_sections[section_index] = None
            plan.dropped_sections.append(dropped["label"])
            current = measure(trimmed_sections)

    plan.evidence_chars_after = sum(
        len(s["text"])
        for s in trimmed_sections
        if s is not None and s["kind"] == "evidence"
    )
    plan.final_tokens = current
    plan.fits_after_user_trim = current <= cap

    if not plan.fits_after_user_trim:
        # everything trimmable is gone; what remains is fixed text + completion
        fixed_user = "".join(
            s["text"] for s in trimmed_sections if s is not None and s["kind"] == "keep"
        )
        fixed_conversation = [dict(conversation[0], content=fixed_user)] + conversation[1:]
        fixed_tokens = count_tokens(fixed_conversation, tokenizer)
        plan.completion_only_tokens = fixed_tokens - plan.completion_tokens
        # could a completion-side cut alone rescue it? only when the fixed
        # prefix itself fits under the cap and the required cut is smaller
        # than the whole completion
        needed = current - cap
        prefix_floor = max(cap - plan.completion_only_tokens, 0)
        max_completion_cut = plan.completion_tokens - prefix_floor
        plan.completion_would_need_trimming = (
            0 < needed <= max_completion_cut
            and plan.completion_only_tokens < cap
        )
        if not plan.completion_would_need_trimming and needed > 0:
            plan.notes.append(
                f"over cap by {needed} tokens with all evidence removed; "
                f"completion is {plan.completion_tokens} tokens "
                f"(fixed prefix alone is {plan.completion_only_tokens})"
            )
    return plan


def analyze(
    tokenizer: Any,
    config: dict[str, Any],
    cap: int,
    splits: list[str] | None = None,
    section_char_budget: int = SECTION_CHAR_BUDGET,
    list_limit: int = 25,
) -> dict[str, Any]:
    """Run the feasibility simulation over all over-cap examples.

    Read-only: returns the report dict; nothing is written to disk here.
    """
    splits = splits or list(SPLIT_FILES)
    per_split: dict[str, Any] = {}
    for split in splits:
        path = FINAL_DIR / f"{split}.jsonl"
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]

        plans: list[TrimPlan] = []
        unrenderable = 0
        for index, example in enumerate(records):
            if extract_conversation(example) is None:
                unrenderable += 1
                continue
            conversation = extract_conversation(example)
            if count_tokens(conversation, tokenizer) <= cap:
                continue
            plan = simulate_trim(
                example, tokenizer, cap, section_char_budget=section_char_budget
            )
            plan.index = index
            plan.split = split
            plans.append(plan)

        outcomes = {"rescue_full": 0, "rescue_partial": 0, "not_rescuable": 0}
        for plan in plans:
            outcomes[plan.outcome] += 1

        def _sorted_examples(outcome: str) -> list[dict[str, Any]]:
            chosen = sorted(
                (p for p in plans if p.outcome == outcome),
                key=lambda p: -p.original_tokens,
            )
            return [p.to_dict() for p in chosen[:list_limit]]

        per_split[split] = {
            "over_cap_examples": len(plans),
            "outcomes": outcomes,
            "rescue_rate_percent": round(
                100 * outcomes["rescue_full"] / len(plans), 2
            )
            if plans
            else 0.0,
            "evidence_removed_tokens_estimate": sum(
                p.original_tokens - p.final_tokens for p in plans
            ),
            "median_evidence_retained_share": _median(
                [p.evidence_retained_share for p in plans]
            ),
            "examples_fully_rescuable": _sorted_examples("rescue_full"),
            "examples_needing_completion_alteration": _sorted_examples("rescue_partial"),
            "examples_not_rescuable": _sorted_examples("not_rescuable"),
            "unrenderable_skipped": unrenderable,
        }

    totals = {
        "over_cap_examples": sum(s["over_cap_examples"] for s in per_split.values()),
        "outcomes": {
            key: sum(s["outcomes"][key] for s in per_split.values())
            for key in ("rescue_full", "rescue_partial", "not_rescuable")
        },
    }
    if totals["over_cap_examples"]:
        totals["rescue_rate_percent"] = round(
            100 * totals["outcomes"]["rescue_full"] / totals["over_cap_examples"], 2
        )
    else:
        totals["rescue_rate_percent"] = 0.0

    return {
        "cap": cap,
        "trim_strategy": {
            "section_char_budget": section_char_budget,
            "trim_priority": list(TRIM_PRIORITY),
            "method": (
                "user-side only: evidence sections head-truncated to the "
                "budget in noisiest-first priority, then least-informative "
                "sections dropped, re-measured through the chat template "
                "after each step; completions never touched"
            ),
        },
        "per_split": per_split,
        "totals": totals,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return round(ordered[mid], 3)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 3)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Trim-feasibility analysis for over-cap examples (read-only)."
    )
    parser.add_argument("--cap", type=int, default=2048, help="max_seq_length to test")
    parser.add_argument("--budget", type=int, default=SECTION_CHAR_BUDGET,
                        help="per-section character budget when trimming")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    args = parser.parse_args(argv)

    from src.data.utils import get_hf_token, load_config

    config = load_config()
    tokenizer_id = config.get("tokenization", {}).get(
        "tokenizer_id", "meta-llama/Llama-3.2-3B-Instruct"
    )
    budget = int(
        config.get("tokenization", {})
        .get("trim_feasibility", {})
        .get("section_char_budget", args.budget)
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=get_hf_token())
    report = analyze(tokenizer, config, cap=args.cap, section_char_budget=budget)

    report_path = FINAL_DIR / "trim_feasibility_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    print(f"\ntrim_feasibility: report saved to {report_path}")
    return 0


def _print_report(report: dict[str, Any]) -> None:
    print(f"Trim Feasibility (cap={report['cap']}, "
          f"budget={report['trim_strategy']['section_char_budget']} chars/section)")
    print("=" * 60)
    for split, info in report["per_split"].items():
        print(f"\n--- {split} " + "-" * (58 - len(split)))
        print(f"  over-cap examples: {info['over_cap_examples']}")
        outcomes = info["outcomes"]
        print(
            f"  rescue_full: {outcomes['rescue_full']}  "
            f"rescue_partial (completion would need cutting): {outcomes['rescue_partial']}  "
            f"not_rescuable: {outcomes['not_rescuable']}"
        )
        print(f"  rescue rate: {info['rescue_rate_percent']}%")
        print(f"  median evidence retained: {info['median_evidence_retained_share']}")
    totals = report["totals"]
    print(f"\nTOTALS: {totals['over_cap_examples']} over-cap examples")
    print(f"  fully rescuable (user-side only): {totals['outcomes']['rescue_full']}")
    print(f"  completion would need cutting:    {totals['outcomes']['rescue_partial']}")
    print(f"  not rescuable:                    {totals['outcomes']['not_rescuable']}")


if __name__ == "__main__":
    raise SystemExit(main())
