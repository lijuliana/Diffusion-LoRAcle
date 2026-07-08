"""Significance machinery shared by every gate (POC-1b wild gate + POC-1c organism gate).

A NeurIPS reviewer will not accept a go/no-go decided by "our_svd mAP beats the baseline by >0.05".
0.05 on a few-hundred-query retrieval set can be noise. So every gate metric here comes with:

  * a PERMUTATION NULL — shuffle the labels many times, recompute the metric, and report the fraction
    of shuffles that match/beat the observed value (an empirical p-value). This is the honest test of
    "is this above chance at all", with no distributional assumption.
  * a BOOTSTRAP CI — resample the per-query average-precisions with replacement to get a confidence
    interval on the mAP, so a margin between two featurizers can be read against its uncertainty.

Both operate on the SAME cosine-from-Gram matrix the probe already builds (kernel space), so nothing
here materializes a feature matrix. Randomness is seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np


def per_query_ap(cos: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[list[float], list[int]]:
    """Return (average_precisions, query_indices) for within-group nearest-neighbour retrieval.

    A query i contributes iff its group has >=2 other members AND relevance is mixed (some same-label
    siblings, some not) — otherwise AP is degenerate/undefined. Identical selection to the probe's
    within_group_retrieval_map, factored out so the null/bootstrap score EXACTLY the same queries.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(y)
    aps: list[float] = []
    qidx: list[int] = []
    for i in range(n):
        sib = np.where((groups == groups[i]) & (np.arange(n) != i))[0]
        if len(sib) < 2:
            continue
        rel = (y[sib] == y[i]).astype(float)
        if rel.sum() == 0 or rel.sum() == len(sib):
            continue
        order = np.argsort(-cos[i, sib])
        rel_sorted = rel[order]
        prec_at = np.cumsum(rel_sorted) / (np.arange(len(rel_sorted)) + 1)
        aps.append(float((prec_at * rel_sorted).sum() / rel_sorted.sum()))
        qidx.append(i)
    return aps, qidx


def _map_from_labels(cos: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    aps, _ = per_query_ap(cos, y, groups)
    return float(np.mean(aps)) if aps else float("nan")


def permutation_pvalue(cos: np.ndarray, y, groups, n_perm: int = 2000, seed: int = 0) -> dict:
    """Empirical p-value for 'within-group retrieval mAP is above chance'.

    Null: labels are exchangeable WITHIN the constraint that group structure is fixed (we permute the
    label vector globally, which is the standard label-shuffle null for 'do features carry label info').
    p = (#{null mAP >= observed} + 1) / (n_perm + 1).  Also returns the null mean/std for context.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    observed = _map_from_labels(cos, y, groups)
    if observed != observed:  # NaN → no valid queries
        return {"observed": None, "p_value": None, "n_perm": 0, "null_mean": None, "n_queries": 0}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        null[t] = _map_from_labels(cos, rng.permutation(y), groups)
    null = null[~np.isnan(null)]
    ge = int(np.sum(null >= observed))
    _, qidx = per_query_ap(cos, y, groups)
    return {
        "observed": round(observed, 4),
        "p_value": round((ge + 1) / (len(null) + 1), 5),
        "n_perm": int(len(null)),
        "null_mean": round(float(np.mean(null)), 4) if len(null) else None,
        "null_std": round(float(np.std(null)), 4) if len(null) else None,
        "n_queries": len(qidx),
    }


def bootstrap_ci(cos: np.ndarray, y, groups, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI on the within-group retrieval mAP (resample the per-query APs)."""
    aps, qidx = per_query_ap(cos, np.asarray(y), np.asarray(groups))
    if not aps:
        return {"map": None, "ci_low": None, "ci_high": None, "n_queries": 0}
    aps = np.asarray(aps)
    rng = np.random.default_rng(seed)
    boots = np.array([aps[rng.integers(0, len(aps), len(aps))].mean() for _ in range(n_boot)])
    return {
        "map": round(float(aps.mean()), 4),
        "ci_low": round(float(np.quantile(boots, alpha / 2)), 4),
        "ci_high": round(float(np.quantile(boots, 1 - alpha / 2)), 4),
        "n_queries": len(aps),
    }


def paired_bootstrap_gt(cos_a: np.ndarray, cos_b: np.ndarray, y, groups,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """Is featurizer A's within-group mAP > featurizer B's, accounting for query-level noise?

    Paired bootstrap over the SHARED query set (only queries valid for both): resample query indices,
    recompute both mAPs on the resample, report P(mAP_A > mAP_B) and the CI on the difference. This is
    how we compare our_svd vs a baseline HONESTLY instead of a hard-coded +0.05 margin.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    aps_a, qa = per_query_ap(cos_a, y, groups)
    aps_b, qb = per_query_ap(cos_b, y, groups)
    shared = sorted(set(qa) & set(qb))
    if not shared:
        return {"delta": None, "p_a_gt_b": None, "ci_low": None, "ci_high": None, "n_queries": 0}
    ia = {q: k for k, q in enumerate(qa)}
    ib = {q: k for k, q in enumerate(qb)}
    a = np.array([aps_a[ia[q]] for q in shared])
    b = np.array([aps_b[ib[q]] for q in shared])
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, len(shared), len(shared))
        deltas[t] = a[idx].mean() - b[idx].mean()
    return {
        "delta": round(float(a.mean() - b.mean()), 4),
        "p_a_gt_b": round(float(np.mean(deltas > 0)), 4),
        "ci_low": round(float(np.quantile(deltas, 0.025)), 4),
        "ci_high": round(float(np.quantile(deltas, 0.975)), 4),
        "n_queries": len(shared),
    }
