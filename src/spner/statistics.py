"""Paired inference with explicit experimental units; standard library only.

F1 inputs use [0, 1]; differences and intervals are exported in percentage points.
These routines do not turn toy fixtures into experimental evidence.
"""
from __future__ import annotations

import itertools
import math
import random
import statistics as st

from .metrics import Counts


def _finite(values):
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("Expected non-empty finite values")
    return result


def percentile(values: list[float], q: float) -> float:
    if not 0 <= q <= 1:
        raise ValueError("Invalid quantile")
    ordered = sorted(_finite(values))
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def sign_flip_test(differences: list[float], *, seed: int = 2026,
                   draws: int = 19999, exact_limit: int = 20) -> dict:
    """Two-sided test of absolute mean paired difference.

    Requires independent paired blocks and exchangeability of method assignments
    under the null (equivalently a suitable sign-symmetry assumption). It is not
    assumption-free. Zero differences do not add permutation information.
    """
    values = _finite(differences)
    nonzero = [value for value in values if value != 0]
    n = len(nonzero)
    if not 0 <= exact_limit <= 22 or draws < 1:
        raise ValueError("Invalid permutation settings")
    observed = abs(math.fsum(nonzero))
    tolerance = 1e-12 * max(1.0, math.fsum(abs(x) for x in nonzero))
    if n <= exact_limit:
        total = 2 ** n
        extreme = sum(abs(math.fsum(s * x for s, x in zip(signs, nonzero))) >= observed - tolerance
                      for signs in itertools.product((-1, 1), repeat=n))
        p_value = extreme / total
        mode = "exact"
    else:
        rng = random.Random(seed)
        extreme = sum(abs(math.fsum(rng.choice((-1, 1)) * x for x in nonzero)) >= observed - tolerance
                      for _ in range(draws))
        total = draws
        p_value = (extreme + 1) / (draws + 1)
        mode = "monte_carlo_plus_one"
    return {"p_value": p_value, "mode": mode, "permutations": total,
            "n_nonzero_pairs": n, "minimum_exact_two_sided_p": min(1.0, 2.0 ** (1 - n))}


def paired_run_summary(a: list[float], b: list[float], *, seed: int = 2026,
                       draws: int = 9999) -> dict:
    """A minus B, resampling paired runs on a fixed test set."""
    left, right = _finite(a), _finite(b)
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Need at least two complete matched pairs")
    if any(not 0 <= x <= 1 for x in left + right):
        raise ValueError("F1 must be in [0, 1], not [0, 100]")
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    differences = [x - y for x, y in zip(left, right)]
    rng, n = random.Random(seed), len(left)
    boot = [st.fmean(rng.choices(differences, k=n)) for _ in range(draws)]
    sd = st.stdev(differences)
    return {"n_pairs": n, "mean_a_f1": st.fmean(left), "mean_b_f1": st.fmean(right),
            "sd_a_f1": st.stdev(left), "sd_b_f1": st.stdev(right),
            "delta_f1_pp": 100 * st.fmean(differences), "sd_delta_pp": 100 * sd,
            "median_delta_pp": 100 * st.median(differences),
            "ci95_low_pp": 100 * percentile(boot, .025),
            "ci95_high_pp": 100 * percentile(boot, .975),
            "paired_dz": st.fmean(differences) / sd if sd > 1e-12 else None,
            "ci_method": "percentile_bootstrap_paired_runs_fixed_test_set",
            "bootstrap_draws": draws, "analysis_seed": seed,
            **sign_flip_test(differences, seed=seed),
            "warnings": (["Fewer than 10 pairs: coarse p-values and unstable intervals; no power guarantee."]
                         if n < 10 else [])}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down FWER adjustment over the complete predeclared family."""
    values = _finite(p_values)
    if any(not 0 <= x <= 1 for x in values):
        raise ValueError("p-values outside [0, 1]")
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    result, previous = [0.0] * len(values), 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        result[index] = previous
    return result


def paired_document_bootstrap(pairs: dict[str, dict[str, tuple[Counts, Counts]]], *,
                              resample_pairs: bool = False, seed: int = 2026,
                              draws: int = 9999) -> dict:
    """Paired document bootstrap, optionally crossed with paired-run resampling.

    Each replicate uses the SAME sampled documents for both methods and every
    selected run. Sum entity counts before computing micro-F1 in each run;
    average those run-level F1 differences. Documents are not nested in seeds.
    Caller must verify identical gold labels, example IDs and scoring policies.
    """
    if not pairs or draws < 100:
        raise ValueError("Need predictions and at least 100 draws")
    pair_ids = sorted(pairs)
    documents = sorted(pairs[pair_ids[0]])
    if len(documents) < 2:
        raise ValueError("Need at least two independent test documents")
    if resample_pairs and len(pair_ids) < 2:
        raise ValueError("Crossed resampling needs at least two independent paired runs")
    support = {}
    for pair_id in pair_ids:
        if set(pairs[pair_id]) != set(documents):
            raise ValueError("All pairs must cover identical test documents")
        for doc in documents:
            a, b = pairs[pair_id][doc]
            if a.tp + a.fn != b.tp + b.fn:
                raise ValueError("Gold entity support differs between methods")
            if doc in support and support[doc] != a.tp + a.fn:
                raise ValueError("Gold entity support differs between runs")
            support[doc] = a.tp + a.fn

    def delta(sampled_pairs, sampled_docs):
        differences = []
        for pair_id in sampled_pairs:
            a = sum((pairs[pair_id][doc][0] for doc in sampled_docs), Counts()).metrics()["f1"]
            b = sum((pairs[pair_id][doc][1] for doc in sampled_docs), Counts()).metrics()["f1"]
            if a is None or b is None:
                raise ValueError("Undefined F1 in bootstrap draw; endpoint too sparse for this interval")
            differences.append(a - b)
        return st.fmean(differences)

    observed = delta(pair_ids, documents)
    rng, boot = random.Random(seed), []
    for _ in range(draws):
        sampled_pairs = rng.choices(pair_ids, k=len(pair_ids)) if resample_pairs else pair_ids
        sampled_docs = rng.choices(documents, k=len(documents))
        boot.append(delta(sampled_pairs, sampled_docs))
    return {"delta_f1_pp": 100 * observed,
            "ci95_low_pp": 100 * percentile(boot, .025),
            "ci95_high_pp": 100 * percentile(boot, .975),
            "n_pairs": len(pair_ids), "n_documents": len(documents),
            "ci_method": ("crossed_paired_runs_documents_percentile" if resample_pairs
                          else "paired_documents_fixed_checkpoints_percentile"),
            "bootstrap_draws": draws, "analysis_seed": seed,
            "warnings": ["Pointwise interval; no multiplicity adjustment.",
                         "Requires independent document clusters and comparable gold annotations."]}
