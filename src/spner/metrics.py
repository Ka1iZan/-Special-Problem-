"""Strict word-span evaluation with auditable masking and overlap partitions."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .data.bio import bio_spans, invalid_i_count, split_label
from .data.overlap import normalize_surface

ENTITY_TYPES = ("PER", "LOC", "ORG")


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.tp, self.fp, self.fn)):
            raise ValueError("Counts must be nonnegative integers")

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def metrics(self) -> dict:
        pden, rden = self.tp + self.fp, self.tp + self.fn
        fden = 2 * self.tp + self.fp + self.fn
        return {**asdict(self), "gold_support": rden, "predicted_support": pden,
                "precision": self.tp / pden if pden else None,
                "recall": self.tp / rden if rden else None,
                "f1": 2 * self.tp / fden if fden else None}


def score_sentence(tokens: list[str], gold: list[str], predicted: list[str], *,
                   source_vocabulary: set[str] | None = None,
                   ignore_policy: str = "reject") -> dict:
    """Mask complete predicted spans touching IGNORE only with explicit opt-in.

    Both gold and predicted spans are partitioned by their OWN surface form.
    This assigns false positives to seen/unseen and keeps counts additive.
    """
    if not len(tokens) == len(gold) == len(predicted):
        raise ValueError("Token/gold/prediction lengths differ")
    if ignore_policy not in {"reject", "exclude_touching_spans"}:
        raise ValueError("Unknown ignore policy")
    for label in gold + predicted:
        _, kind = split_label(label)
        if kind is not None and kind not in ENTITY_TYPES:
            raise ValueError(f"Unknown entity type: {kind}")
    if "IGNORE" in predicted:
        raise ValueError("IGNORE is a gold annotation mask, never a prediction")
    masked = {i for i, label in enumerate(gold) if label == "IGNORE"}
    if masked and ignore_policy == "reject":
        raise ValueError("IGNORE present: explicitly choose an audited evaluation policy")
    gold_spans = set(bio_spans(gold))
    predicted_raw = set(bio_spans(predicted, invalid="discard"))
    pred_spans = {span for span in predicted_raw
                  if not masked.intersection(range(span.start, span.end))}

    def counts(g, p):
        return Counts(len(g & p), len(p - g), len(g - p)).metrics()

    results = {"all": counts(gold_spans, pred_spans), "by_type": {
        kind: counts({s for s in gold_spans if s.label == kind},
                     {s for s in pred_spans if s.label == kind}) for kind in ENTITY_TYPES},
        "ignored_tokens": len(masked),
        "excluded_predicted_spans": len(predicted_raw - pred_spans),
        "invalid_predicted_i_tags": invalid_i_count(predicted),
        "ignore_policy": ignore_policy,
        "bio_policy": "strict_iob2_discard_invalid_i"}
    if source_vocabulary is not None:
        def seen(span):
            return normalize_surface(tokens[span.start:span.end]) in source_vocabulary
        results["by_overlap"] = {
            name: counts({s for s in gold_spans if seen(s) == flag},
                         {s for s in pred_spans if seen(s) == flag})
            for name, flag in (("seen", True), ("unseen", False))}
    return results


def aggregate_scores(scores: list[dict]) -> dict:
    """Sum entity counts, never average sentence F1."""
    if not scores:
        raise ValueError("Cannot aggregate an empty evaluation set")
    for key in ("ignore_policy", "bio_policy"):
        if len({score[key] for score in scores}) != 1:
            raise ValueError(f"Mixed {key} values")
    if len({"by_overlap" in score for score in scores}) != 1:
        raise ValueError("Mixed overlap vocabulary availability")

    def total(rows):
        return sum((Counts(row["tp"], row["fp"], row["fn"]) for row in rows), Counts()).metrics()

    by_type = {kind: total([score["by_type"][kind] for score in scores])
               for kind in ENTITY_TYPES}
    class_f1 = [value["f1"] for value in by_type.values()]
    result = {"all": total([s["all"] for s in scores]), "by_type": by_type,
              "macro_f1": sum(class_f1) / 3 if all(x is not None for x in class_f1) else None,
              "macro_policy": "three_fixed_classes; undefined_if_any_class_undefined",
              "ignore_policy": scores[0]["ignore_policy"], "bio_policy": scores[0]["bio_policy"]}
    for key in ("ignored_tokens", "excluded_predicted_spans", "invalid_predicted_i_tags"):
        result[key] = sum(s[key] for s in scores)
    if "by_overlap" in scores[0]:
        result["by_overlap"] = {scope: total([s["by_overlap"][scope] for s in scores])
                                for scope in ("seen", "unseen")}
    return result


def forgetting(f1_before: float, f1_after: float) -> float:
    return float(f1_before - f1_after)


def transfer_gain(source_initialized_f1: float, target_only_f1: float) -> float:
    return float(source_initialized_f1 - target_only_f1)
