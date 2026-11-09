from __future__ import annotations

import unicodedata

from .bio import bio_spans
from .schema import NERSentence


def normalize_surface(tokens: list[str]) -> str:
    """Diagnostic normalization only: NFC, casefold, whitespace; retain punctuation."""
    return " ".join(unicodedata.normalize("NFC", " ".join(tokens)).casefold().split())


def training_entity_vocabulary(records: list[NERSentence]) -> set[str]:
    if any(record.split != "train" for record in records):
        raise ValueError("Entity-overlap vocabulary must use source training only")
    return {normalize_surface(record.tokens[span.start:span.end])
            for record in records for span in bio_spans(record.labels_coarse)}
