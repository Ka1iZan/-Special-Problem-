from __future__ import annotations
from dataclasses import dataclass
import random
import math

@dataclass
class MemoryItem:
    example_id: str
    language: str
    entity_types: frozenset[str]

class ReplayIndex:
    def __init__(self, items: list[MemoryItem]):
        if not items or len({item.example_id for item in items}) != len(items):
            raise ValueError("Memory must contain distinct non-empty items")
        if any(not item.example_id or item.language not in {"tag", "ceb", "hil"}
               or not item.entity_types <= {"PER", "LOC", "ORG"} for item in items):
            raise ValueError("Invalid memory item")
        self.items = {x.example_id: x for x in items}
        self.by_group: dict[tuple[str, str], list[str]] = {}
        for item in items:
            for entity_type in sorted(item.entity_types):
                self.by_group.setdefault((item.language, entity_type), []).append(item.example_id)

    def sample_from_group(self, group: tuple[str, str], rng: random.Random) -> MemoryItem:
        ids = self.by_group.get(group, [])
        if not ids:
            raise KeyError(f"Empty replay group: {group}")
        return self.items[rng.choice(ids)]

    def sample_uniform_sentence(self, rng: random.Random) -> MemoryItem:
        return self.items[rng.choice(sorted(self.items))]

    def group_probabilities(self, drifts: dict[tuple[str, str], float] | None = None,
                            tau: float = 0.5) -> dict[tuple[str, str], float]:
        """Uniform groups is the control for drift priorities; not uniform sentences."""
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        groups = sorted(self.by_group)
        if not groups:
            raise ValueError("Memory has no entity groups")
        if drifts is not None and set(drifts) != set(groups):
            raise ValueError("Drift values must cover exactly the available memory groups")
        values = [0.0 if drifts is None else drifts[group] for group in groups]
        if any(not math.isfinite(value) or not 0 <= value <= 2 for value in values):
            raise ValueError("Cosine drift must be finite and in [0, 2]")
        maximum = max(values)
        weights = [math.exp((value - maximum) / tau) for value in values]
        total = sum(weights)
        return dict(zip(groups, [w / total for w in weights]))

    def sample_group(self, rng: random.Random, *,
                     drifts: dict[tuple[str, str], float] | None = None,
                     tau: float = 0.5) -> MemoryItem:
        probabilities = self.group_probabilities(drifts, tau)
        group = rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]
        return self.sample_from_group(group, rng)
