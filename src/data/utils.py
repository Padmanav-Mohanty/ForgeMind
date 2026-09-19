"""Shared utilities for the ForgeMind data pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_ROLES = {"system", "user", "assistant"}

# Controlled task vocabulary for metadata["task"]. Converters must never
# force an example into a category the source does not support.
TASK_VOCABULARY = {
    "debugging",
    "root_cause_analysis",
    "bug_fix",
    "error_analysis",
    "code_explanation",
    "test_generation",
    "regression_testing",
    "issue_resolution",
    "dependency_error",
    "runtime_error",
    "configuration_error",
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load configs/data_config.yaml."""
    import yaml

    config_path = path or PROJECT_ROOT / "configs" / "data_config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines. Malformed lines raise."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Write records as JSONL atomically enough for our purposes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
            count += 1
    tmp_path.replace(path)
    return count


def read_text(path: Path, max_bytes: int | None = None) -> str | None:
    """Read a text file defensively. Returns None when unreadable."""
    try:
        if not path.is_file():
            return None
        if max_bytes is not None and path.stat().st_size > max_bytes:
            with open(path, "rb") as fh:
                return fh.read(max_bytes).decode("utf-8", errors="replace")
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_json(path: Path) -> Any:
    """Read a JSON file defensively. Returns None when unreadable/invalid."""
    text = read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# Message helpers

def user_message(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


def build_example(
    user_content: str,
    assistant_content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build an example following the common ForgeMind schema.

    Metadata keys with empty/None values are omitted rather than invented.
    """
    cleaned_metadata = {
        key: value for key, value in metadata.items() if value not in (None, "")
    }
    return {
        "messages": [user_message(user_content), assistant_message(assistant_content)],
        "metadata": cleaned_metadata,
    }


def get_user_content(example: dict[str, Any]) -> str:
    """Return concatenated user content of an example ('' when absent)."""
    parts = [
        message.get("content", "")
        for message in example.get("messages", [])
        if message.get("role") == "user"
    ]
    return "\n".join(part for part in parts if part)


def get_assistant_content(example: dict[str, Any]) -> str:
    """Return concatenated assistant content of an example ('' when absent)."""
    parts = [
        message.get("content", "")
        for message in example.get("messages", [])
        if message.get("role") == "assistant"
    ]
    return "\n".join(part for part in parts if part)


def get_group_key(example: dict[str, Any]) -> str | None:
    """Leakage-prevention group key for splitting (None when absent)."""
    metadata = example.get("metadata") or {}
    return metadata.get("group_key")


def contains_code(text: str) -> bool:
    """Heuristic: does this text contain fenced or indented code blocks?"""
    if "```" in text:
        return True
    # Indented code block: a line starting with 4 spaces or a tab
    return bool(re.search(r"^(?:    |\t)\S", text, flags=re.MULTILINE))


# Hashing / dedup helpers

def normalize_for_hash(text: str) -> str:
    """Deterministic text normalization for hashing/dedup.

    Collapses whitespace variations so that examples differing only in
    formatting hash identically.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip().lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_fingerprint(example: dict[str, Any]) -> str:
    """SHA-256 of normalized user+assistant content."""
    combined = (
        normalize_for_hash(get_user_content(example))
        + "\n\x00\n"
        + normalize_for_hash(get_assistant_content(example))
    )
    return sha256_text(combined)


def token_set(text: str) -> set[str]:
    """Cheap word-token set for near-duplicate detection."""
    return set(re.findall(r"[a-z0-9_]+", normalize_for_hash(text).lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


# GitHub helpers

def github_repo_id_to_slug(dataset_dir: Path, repo_id: int) -> str | None:
    """Resolve a GHPR numeric repo_id to 'owner/name' via the manifest."""
    manifest = read_json(dataset_dir / "repos" / "_repo_ids.json")
    if isinstance(manifest, dict):
        return manifest.get(str(repo_id))
    return None


def strip_html_tags(text: str) -> str:
    """Very small helper for trimming HTML snippets in issue bodies."""
    return re.sub(r"<[^>]+>", " ", text)


# HF token handling (never print or log the token itself)

def get_hf_token() -> str | None:
    """Return a Hugging Face token from env/.env when configured."""
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("[") or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip().upper() in {"HF_TOKEN", "HUGGINGFACE_TOKEN"}:
                        return value.strip()
        except OSError:
            return None
    return None


def get_github_token() -> str | None:
    """Return a GitHub token from the environment when configured.

    Deliberately does NOT read .env: a Hugging Face token stored there is
    not valid for GitHub API authentication.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return token.strip() if token else None
