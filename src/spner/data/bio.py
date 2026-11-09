"""Word-offset BIO handling. Offsets are zero-based, end-exclusive."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    label: str


def split_label(label: str) -> tuple[str, str | None]:
    if label in {"O", "IGNORE"}:
        return label, None
    if not isinstance(label, str) or "-" not in label:
        raise ValueError(f"Invalid BIO label: {label!r}")
    prefix, kind = label.split("-", 1)
    if prefix not in {"B", "I"} or not kind:
        raise ValueError(f"Invalid BIO label: {label!r}")
    return prefix, kind


def bio_spans(labels: Iterable[str], *, invalid: str = "error") -> list[Span]:
    """Gold: reject illegal I tags. Predictions: optionally discard invalid I runs.

    'discard' requires a B tag to start a span (strict IOB2 extraction).
    'repair' is an explicit alternative and must not be silently used for scoring.
    IGNORE is a boundary, never a token to remove and concatenate across.
    """
    if invalid not in {"error", "discard", "repair"}:
        raise ValueError(f"Unknown BIO policy: {invalid}")
    tags = list(labels)
    spans: list[Span] = []
    start, active = None, None
    for index, label in enumerate(tags + ["O"]):
        prefix, kind = split_label(label)
        continuation = prefix == "I" and active == kind
        if active is not None and not continuation:
            spans.append(Span(start, index, active))
            start, active = None, None
        if prefix == "B":
            start, active = index, kind
        elif prefix == "I" and not continuation:
            if invalid == "error":
                raise ValueError(f"Invalid IOB2 transition at word {index}: {label}")
            if invalid == "repair":
                start, active = index, kind
    return spans


def invalid_i_count(labels: Iterable[str]) -> int:
    active, count = None, 0
    for label in labels:
        prefix, kind = split_label(label)
        if prefix == "B":
            active = kind
        elif prefix == "I":
            if active != kind:
                count += 1
                active = None
        else:
            active = None
    return count
