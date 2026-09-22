"""Rendering helpers: ForgeMind JSONL examples -> tokenizable training texts.

The number that matters for `max_seq_length` is the token count of the text
the training stage will actually tokenize. TRL's supervised fine-tuning
workflow renders conversations through the tokenizer's own chat template
before tokenizing, so that is exactly what these helpers reproduce — no
hand-rolled prompt format is invented here.

Design note: everything in this module is pure and tokenizer-agnostic.
Functions take any object exposing the two tokenizer callables they need
(`apply_chat_template`, and `__call__`/`encode` for text->ids), so the core
logic is unit-testable with a stub tokenizer and no model downloads. The
Hugging Face tokenizer itself is only instantiated in `tokenize_dataset`.

Counting convention (documented for the report):

  An example's token length is
      len(tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=False))
  i.e. BOS + full chat-template rendering of the conversation: per-message
  `<|start_header_id|>role<|end_header_id|>` framing, the template's default
  dated system preamble, every content token, and one <|eot_id|> per message.
  Prompt-only lengths (everything up to and including the final assistant
  header) are also recorded so completion vs prompt share can be reported.

  Statistics are computed once per example over the train split — examples
  are never truncated before measurement, and the input JSONL is never
  modified.
"""

from __future__ import annotations

import math
from typing import Any

from src.data.utils import VALID_ROLES


def validate_messages(example: dict[str, Any]) -> list[str]:
    """Return a list of structural problems with an example's messages.

    Empty list means the example can be rendered for tokenization.
    Never raises: invalid examples are reported, not crashed on.
    """
    problems: list[str] = []
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["messages_missing_or_empty"]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(f"message[{index}]_not_an_object")
            continue
        role = message.get("role")
        if role not in VALID_ROLES:
            problems.append(f"message[{index}]_invalid_role")
            continue
        content = message.get("content")
        if not isinstance(content, str):
            problems.append(f"message[{index}]_content_not_a_string")
        elif not content.strip():
            problems.append(f"message[{index}]_empty_content")
    return problems


def extract_conversation(example: dict[str, Any]) -> list[dict[str, str]] | None:
    """Extract [{role, content}, ...] from an example, or None if unusable.

    Keeps only system/user/assistant messages with string content — the
    exact structure chat templates expect.
    """
    if validate_messages(example):
        return None
    return [
        {"role": message["role"], "content": message["content"]}
        for message in example["messages"]
    ]


def render_for_training(
    conversation: list[dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> str:
    """Render a conversation to its training-time text via the chat template.

    Raises ValueError when the tokenizer has no chat template rather than
    silently falling back to concatenation (a wrong format would make every
    measured length wrong in a way the report can't see).
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "tokenizer has no chat template; cannot reproduce the training "
            "rendering. Use an instruct tokenizer that matches the model "
            "being fine-tuned."
        )
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def count_tokens(
    conversation: list[dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> int:
    """Token length of an example under the training chat format.

    Uses the template's own tokenization when available (authoritative —
    this is byte-for-byte what TRL will feed the model), and falls back to
    tokenizing the rendered string with special tokens included.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "tokenizer has no chat template; cannot reproduce the training "
            "rendering. Use an instruct tokenizer that matches the model "
            "being fine-tuned."
        )
    tokenized = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    # transformers >= 5 returns a BatchEncoding here; older versions and
    # plain-list stubs return the id list directly.
    if hasattr(tokenized, "input_ids"):
        input_ids = tokenized.input_ids
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)
    return len(tokenized)


def completion_token_span(conversation: list[dict[str, str]], tokenizer: Any) -> int:
    """Tokens attributable to the final assistant turn, including its framing.

    Difference between the full rendering and the rendering of everything
    before the final message plus the generation prompt — i.e. what a
    completion-only loss mask would train on.

    Note: add_generation_prompt on the *full* conversation is deliberately
    NOT used; chat templates only emit the generation prompt after a user
    turn, so it would be a no-op here.
    """
    if len(conversation) < 2:
        return 0
    full = count_tokens(conversation, tokenizer, add_generation_prompt=False)
    prompt_only = count_tokens(
        conversation[:-1], tokenizer, add_generation_prompt=True
    )
    return max(full - prompt_only, 0)


def percentile(sorted_values: list[int], pct: float) -> int:
    """Nearest-rank percentile of an already-sorted list of ints.

    Nearest-rank: the smallest value whose cumulative rank is >= pct% of the
    data (ceil(n * pct / 100)). For a 10-element list, p95 is the maximum —
    interpolation (as numpy's default) is deliberately not used so reported
    values are always observed lengths, never invented ones.
    """
    if not sorted_values:
        return 0
    if not 0.0 < pct <= 100.0:
        raise ValueError(f"pct must be in (0, 100], got {pct}")
    rank = max(1, math.ceil(len(sorted_values) * pct / 100))
    return sorted_values[min(rank, len(sorted_values)) - 1]
