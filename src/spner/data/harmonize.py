from __future__ import annotations

from .bio import bio_spans

VALID_COARSE = {"PER", "LOC", "ORG", "IGNORE"}

def project_bio_label(label: str, mapping: dict[str, str]) -> str:
    """Project a fine BIO label into PER/LOC/ORG while preserving B/I.

    Returns "IGNORE" for unsupported supervised entity types.
    """
    if label == "O":
        return "O"
    try:
        prefix, entity_type = label.split("-", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid BIO label: {label}") from exc
    if prefix not in {"B", "I"}:
        raise ValueError(f"Unsupported BIO prefix: {prefix}")
    mapped = mapping.get(entity_type)
    if mapped is None:
        raise KeyError(f"No projection rule for fine label {entity_type!r}")
    if mapped not in VALID_COARSE:
        raise ValueError(f"Invalid projected entity type: {mapped}")
    return "IGNORE" if mapped == "IGNORE" else f"{prefix}-{mapped}"


def project_sequence(labels: list[str], mapping: dict[str, str]) -> list[str]:
    """Reject malformed input before projection can hide annotation errors."""
    bio_spans(labels)
    result = [project_bio_label(label, mapping) for label in labels]
    bio_spans(result)
    return result
