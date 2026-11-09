"""Dependency-free alignment of a fast tokenizer's word_ids() output."""
from __future__ import annotations

LABELS = ("O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def align_first_subwords(word_ids: list[int | None], labels: list[str]) -> dict:
    """Reject missing/reordered words; caller must not silently truncate examples.

    Obtain word_ids with truncation=False. This checks word coverage, but cannot
    detect loss of continuation pieces of the final word in a pre-truncated input.
    """
    encoded, first_positions = [], {}
    previous = None
    for position, word in enumerate(word_ids):
        if word is None:
            encoded.append(-100)
            previous = None
            continue
        if not isinstance(word, int) or not 0 <= word < len(labels):
            raise ValueError(f"Invalid word index: {word}")
        if word == previous:
            encoded.append(-100)
        else:
            if word in first_positions or word != len(first_positions):
                raise ValueError("Non-contiguous or reordered word mapping")
            label = labels[word]
            if label != "IGNORE" and label not in LABEL_TO_ID:
                raise ValueError(f"Invalid coarse label: {label}")
            encoded.append(-100 if label == "IGNORE" else LABEL_TO_ID[label])
            first_positions[word] = position
        previous = word
    if len(first_positions) != len(labels):
        raise ValueError("Missing words: possible truncation; window explicitly before alignment")
    return {"labels": encoded, "first_subword_positions": list(first_positions.values())}
