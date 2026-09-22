"""Tokenizer testing and sequence-length analysis for ForgeMind.

The analysis stage between dataset preparation and QLoRA training: measures
token-length distributions of the final dataset under the *training* chat
format and evaluates candidate `max_seq_length` caps.

Modules:
    render           — pure helpers that turn JSONL examples into the exact
                       tokenizable texts training will use (unit-testable
                       with a stub tokenizer; no Hugging Face download needed)
    tokenize_dataset — loads the real tokenizer, measures the final dataset,
                       and writes the token-length / cap-analysis report

Usage:
    python -m src.tokenization.tokenize_dataset
    scripts/analyze_tokens.py
"""

from __future__ import annotations
