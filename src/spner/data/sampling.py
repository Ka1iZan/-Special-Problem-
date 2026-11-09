from __future__ import annotations

import hashlib
import math
import random

from .io import validate_unique
from .schema import NERSentence


def nested_target_subsets(records: list[NERSentence], budgets: list[float],
                          seed: int) -> dict[str, dict]:
    """One shuffled training order per seed, then nested sentence-count prefixes.

    This is random sentence sampling, NOT entity-stratified or document sampling.
    No retries to guarantee class coverage; report missing classes as observed.
    """
    validate_unique(records)
    if any(row.split != "train" for row in records):
        raise ValueError("Subset selection is restricted to training records")
    if len({row.language for row in records}) != 1:
        raise ValueError("Select one target language at a time")
    if not budgets or any(not math.isfinite(b) or not 0 < b <= 100 for b in budgets):
        raise ValueError("Budgets must be finite percentages in (0, 100]")
    if len(set(budgets)) != len(budgets):
        raise ValueError("Duplicate budgets")
    ids = sorted(row.example_id for row in records)
    random.Random(seed).shuffle(ids)
    results = {}
    for budget in sorted(budgets):
        size = max(1, math.ceil(len(ids) * budget / 100))
        selected = sorted(ids[:size])
        digest = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
        results[f"{budget:g}"] = {"subset_seed": seed, "requested_percent": budget,
                                  "actual_percent": 100 * size / len(ids),
                                  "n_sentences": size, "subset_id": digest,
                                  "example_ids": selected}
    return results
