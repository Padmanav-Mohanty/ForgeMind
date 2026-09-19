"""
Test suite for the ForgeMind data pipeline (cleaning -> dedup -> normalize -> split).

How to run:
    pytest tests/data -v

Note: the schema/leakage/split tests only run once data/final actually exists
(they auto-skip on a fresh clone, before `python -m src.data.prepare` has been run).
Everything else (cleaning, dedup, normalize, split determinism) builds its own
fixtures in tmp_path so it always runs, even with no data downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.clean import _is_placeholder, _validate_structure, clean_file
from src.data.deduplicate import deduplicate_records
from src.data.normalize import normalize_text
from src.data.split import split_files
from src.data.utils import (
    PROJECT_ROOT,
    build_example,
    content_fingerprint,
    normalize_for_hash,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

# the three raw sources we pull from
SOURCE_NAMES = ("aifaultbench", "ghpr", "defects4j")


def _read_jsonl(path: Path) -> list[dict]:
    """Small helper — read a jsonl file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _final_file(split: str) -> Path:
    return FINAL_DIR / f"{split}.jsonl"


def _make_example(user: str, assistant: str, group_key: str | None = None, source: str = "test") -> dict:
    """Build a minimal fixture example without touching real data."""
    metadata = {"source": source, "task": "debugging", "language": "python"}
    if group_key:
        metadata["group_key"] = group_key
    return build_example(user, assistant, metadata)


# Schema tests — only make sense once data/final has actually been generated

@pytest.fixture(scope="module")
def final_available() -> bool:
    return _final_file("train").is_file() and _final_file("test").is_file()


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "final" / "train.jsonl").is_file(),
    reason="data/final not generated yet (run python -m src.data.prepare)",
)
class TestFinalSchema:
    @pytest.fixture(scope="class")
    def all_final_records(self) -> list[dict]:
        records = []
        for split in ("train", "validation", "test"):
            records.extend(_read_jsonl(_final_file(split)))
        return records

    def test_every_line_is_valid_json(self):
        # _read_jsonl blows up on malformed json, so getting past this
        # loop without an exception is itself the assertion
        for split in ("train", "validation", "test"):
            assert len(_read_jsonl(_final_file(split))) > 0

    def test_every_example_has_messages(self, all_final_records):
        assert all(isinstance(r.get("messages"), list) and r["messages"] for r in all_final_records)

    def test_roles_are_valid(self, all_final_records):
        for record in all_final_records:
            for message in record["messages"]:
                assert message["role"] in {"system", "user", "assistant"}

    def test_user_and_assistant_exist(self, all_final_records):
        for record in all_final_records:
            roles = {m["role"] for m in record["messages"]}
            assert "user" in roles and "assistant" in roles

    def test_metadata_is_object_with_source(self, all_final_records):
        for record in all_final_records:
            metadata = record.get("metadata")
            assert isinstance(metadata, dict)
            assert metadata.get("source") in SOURCE_NAMES

    def test_task_vocabulary_respected(self, all_final_records):
        from src.data.utils import TASK_VOCABULARY

        for record in all_final_records:
            task = record["metadata"].get("task")
            assert task in TASK_VOCABULARY

    def test_no_exact_duplicates_remain(self, all_final_records):
        fingerprints = [content_fingerprint(r) for r in all_final_records]
        assert len(fingerprints) == len(set(fingerprints))

    def test_split_files_exist_with_expected_names(self):
        for split in ("train", "validation", "test"):
            assert _final_file(split).is_file()

    def test_split_sizes_match_report(self, all_final_records):
        report = json.loads((FINAL_DIR / "split_report.json").read_text(encoding="utf-8"))
        total = report["total"]
        assert len(all_final_records) == total
        for split in ("train", "validation", "test"):
            info = report["splits"][split]
            assert len(_read_jsonl(_final_file(split))) == info["examples"]
            assert abs(info["share"] - info["examples"] / total) < 1e-3

    def test_no_group_leakage_across_splits(self):
        # same repo/PR should never end up on both sides of a split
        from src.data.utils import get_group_key

        by_split = {}
        for split in ("train", "validation", "test"):
            by_split[split] = {
                get_group_key(r) for r in _read_jsonl(_final_file(split)) if get_group_key(r)
            }
        names = list(by_split)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlap = by_split[left] & by_split[right]
                assert not overlap, f"group leakage between {left} and {right}: {list(overlap)[:3]}"

    def test_no_fabricated_empty_groups(self, all_final_records):
        # anything coming from a grouped source must carry a group_key,
        # otherwise the leakage check above is silently toothless
        for record in all_final_records:
            source = record["metadata"]["source"]
            if source in SOURCE_NAMES:
                assert record["metadata"].get("group_key"), f"missing group_key in {source} record"


# Cleaning

class TestCleaning:
    def test_removes_missing_assistant(self, tmp_path):
        records = [
            {"messages": [{"role": "user", "content": "hello"}], "metadata": {}},
            _make_example("a" * 50, "b" * 50),
        ]
        path = tmp_path / "x.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        stats = clean_file(path, {"cleaning": {"min_input_length": 20, "min_output_length": 20}})
        assert stats["after"] == 1
        assert stats["removal_reasons"].get("missing_assistant") == 1

    def test_removes_placeholder_but_keeps_body_mentioning_none(self):
        # "None" as a lone token should get flagged, but a real sentence
        # that happens to contain the word "None" should survive
        assert _is_placeholder("None")
        assert _is_placeholder("TODO")
        assert _is_placeholder("...")
        assert not _is_placeholder("The API returns None when the cache is cold. " * 3)

    def test_short_examples_with_code_are_kept(self, tmp_path):
        # code blocks get an exemption from the min-length filter
        code_example = _make_example("Failing code:\n```python\nx = 1\n```", "The call `x = 1` fails.")
        path = tmp_path / "code.jsonl"
        path.write_text(json.dumps(code_example), encoding="utf-8")
        stats = clean_file(path, {"cleaning": {"min_input_length": 20, "min_output_length": 20}})
        assert stats["after"] == 1

    def test_short_plain_example_removed(self, tmp_path):
        path = tmp_path / "short.jsonl"
        path.write_text(json.dumps(_make_example("too short", "me too")), encoding="utf-8")
        stats = clean_file(path, {"cleaning": {"min_input_length": 20, "min_output_length": 20}})
        assert stats["after"] == 0

    def test_structure_validation_reasons(self):
        assert _validate_structure({"messages": []}) == "missing_messages"
        assert _validate_structure({"messages": [{"role": "bot", "content": "x"}]}) == "invalid_role"
        assert _validate_structure({"messages": [{"role": "user", "content": "x"}]}) == "missing_assistant"
        assert _validate_structure("nope") == "not_an_object"


# Deduplication

class TestDeduplication:
    def test_exact_duplicates_removed(self):
        ex1 = _make_example("Identical user prompt " * 3, "Identical answer " * 3)
        # ex2 is the same content, just with different case/whitespace
        ex2 = _make_example(
            "identical  USER   prompt  " + "identical user prompt " * 2,
            "identical\tANSWER  " + "identical answer " * 2,
        )
        kept, stats = deduplicate_records([ex1, ex2], near=False)
        assert stats["exact_removed"] == 1
        assert len(kept) == 1

    def test_formatting_only_difference_hashes_equal(self):
        a = _make_example("Line one\nLine two  ", "The  ANSWER\n\nwith spacing")
        b = _make_example("line ONE\nline TWO", "the answer\n\nwith   spacing")
        assert content_fingerprint(a) == content_fingerprint(b)
        # but a real structural difference (blank line removed) should not hash equal
        c = _make_example("Line one\nLine two", "the answer\nwith spacing")
        assert content_fingerprint(a) != content_fingerprint(c)

    def test_near_duplicates_removed(self):
        base_user = "This is a fairly long user prompt about a NullPointerException in the parser module. " * 2
        base_assistant = "### Root Cause\nThe parser dereferences a null config object in module init. " * 2
        ex1 = _make_example(base_user, base_assistant, group_key="g1")
        ex2 = _make_example(base_user + " tiny variation", base_assistant, group_key="g2")
        kept, stats = deduplicate_records([ex1, ex2], near=True, near_threshold=0.85)
        assert stats["near_removed"] == 1

    def test_distinct_examples_kept(self):
        ex1 = _make_example(
            "A prompt about dependency version conflicts in Python packaging",
            "Answer one about pip resolution order in depth.",
        )
        ex2 = _make_example(
            "A report about GPU memory exhaustion during large batch training",
            "Answer two about gradient checkpointing tradeoffs.",
        )
        kept, stats = deduplicate_records([ex1, ex2], near=True, near_threshold=0.85)
        assert stats == {"exact_removed": 0, "near_removed": 0}
        assert len(kept) == 2

    def test_dedup_is_deterministic(self):
        examples = [
            _make_example(f"unique prompt {i} " + "x " * 30, f"answer {i} " + "y " * 30, group_key=f"g{i}")
            for i in range(5)
        ]
        examples.append(
            _make_example(examples[2]["messages"][0]["content"], examples[2]["messages"][1]["content"], group_key="dup")
        )
        r1 = deduplicate_records(list(examples), near=True)
        r2 = deduplicate_records(list(examples), near=True)
        assert r1 == r2


# Normalization

class TestNormalization:
    def test_crlf_and_trailing_whitespace(self):
        text = "Line one   \r\nLine two\t \r\n\r\n\r\n\r\nLine three"
        out = normalize_text(text)
        assert "\r" not in out
        assert "Line one\nLine two\n\n\nLine three" == out or "Line one\nLine two\n\nLine three" == out

    def test_code_indentation_preserved(self):
        # anything inside a fenced code block should come out byte-identical
        text = "```python\ndef f():\n    return 1\n```"
        assert normalize_text(text) == text

    def test_outside_fence_leading_blank_lines_removed_but_indent_kept(self):
        # leading blank lines get trimmed, but indentation on real content stays
        text = "\n\n    indented line"
        assert normalize_text(text) == "    indented line"
        assert normalize_text("Keep  spaces\n    next indent") == "Keep  spaces\n    next indent"


# Splitting — determinism + no group leakage

class TestSplitting:
    @pytest.fixture()
    def processed_dir(self, tmp_path, monkeypatch):
        import src.data.split as split_module

        records = []
        # 20 grouped examples (one group per record) + 10 ungrouped singletons
        for i in range(20):
            records.append(
                _make_example(f"user content for group {i} " * 3, f"assistant answer {i} " * 3, group_key=f"proj#{i}")
            )
        for i in range(10):
            records.append(_make_example(f"singleton example {i} " * 3, f"singleton answer {i} " * 3))

        for name in ("aifaultbench", "ghpr", "defects4j"):
            (tmp_path / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records), encoding="utf-8"
            )
        monkeypatch.setattr(split_module, "PROCESSED_DIR", tmp_path)
        return split_module

    def _config(self):
        return {"seed": 42, "split": {"train": 0.8, "validation": 0.1, "test": 0.1}}

    def test_split_produces_all_files_and_ratios(self, processed_dir, tmp_path):
        out = tmp_path / "final"
        report = processed_dir.split_files(self._config(), output_dir=out)
        assert set(report["splits"]) == {"train", "validation", "test"}
        sizes = [report["splits"][s]["examples"] for s in ("train", "validation", "test")]
        total = sum(sizes)
        assert abs(sizes[0] / total - 0.8) < 0.05
        assert abs(sizes[1] / total - 0.1) < 0.03
        assert abs(sizes[2] / total - 0.1) < 0.03
        assert (out / "train.jsonl").is_file()

    def test_same_seed_reproducible(self, processed_dir, tmp_path):
        out1, out2 = tmp_path / "a", tmp_path / "b"
        processed_dir.split_files(self._config(), output_dir=out1)
        processed_dir.split_files(self._config(), output_dir=out2)
        for split in ("train", "validation", "test"):
            assert (out1 / f"{split}.jsonl").read_bytes() == (out2 / f"{split}.jsonl").read_bytes()

    def test_different_seed_changes_assignment(self, processed_dir, tmp_path):
        out1 = tmp_path / "s42"
        out2 = tmp_path / "s7"
        processed_dir.split_files(self._config(), output_dir=out1)
        cfg = self._config()
        cfg["seed"] = 7
        processed_dir.split_files(cfg, output_dir=out2)
        assert (out1 / "train.jsonl").read_bytes() != (out2 / "train.jsonl").read_bytes()

    def test_groups_never_split(self, processed_dir, tmp_path):
        out = tmp_path / "final"
        processed_dir.split_files(self._config(), output_dir=out)
        group_to_splits: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            for record in _read_jsonl(out / f"{split}.jsonl"):
                key = record["metadata"].get("group_key")
                if key:
                    group_to_splits.setdefault(key, set()).add(split)
        offenders = {k: v for k, v in group_to_splits.items() if len(v) > 1}
        assert not offenders, f"groups scattered across splits: {offenders}"

    def test_ratios_must_sum_to_one(self, processed_dir, tmp_path):
        bad = {"seed": 1, "split": {"train": 0.5, "validation": 0.2, "test": 0.1}}
        with pytest.raises(ValueError):
            processed_dir.split_files(bad, output_dir=tmp_path / "x")


# Sanity check — raw/ must stay untouched by the pipeline

class TestRawImmutability:
    def test_no_processed_outputs_inside_raw(self):
        raw = PROJECT_ROOT / "data" / "raw"
        if not raw.is_dir():
            pytest.skip("no raw data downloaded")
        derived = [
            p for p in raw.rglob("*.jsonl")
            if ".cache" not in p.parts and p.name not in {"dataset.jsonl"}  # legacy scaffold file, not a pipeline output
        ]
        assert derived == [], f"derived files leaked into raw: {derived[:3]}"