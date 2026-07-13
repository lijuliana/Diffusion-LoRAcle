"""Tests for the gate significance machinery (permutation null + bootstrap). These guard the code that
now DECIDES the go/no-go, so a bug here would silently corrupt every gate verdict.

Constructs cosine matrices with KNOWN structure:
  * a signal matrix where within-group nearest neighbours share the label → mAP high, perm p small;
  * a noise matrix (random) → mAP ≈ chance, perm p large (not significant);
and checks the paired bootstrap prefers signal over noise.
"""

from __future__ import annotations

import numpy as np

from ditloracle.probe.significance import (
    bootstrap_ci,
    paired_bootstrap_gt,
    permutation_pvalue,
    per_query_ap,
)


def _setup(n_per_class=8, n_classes=3, seed=0):
    """One group, balanced classes. Signal cos = block structure (same-class similar); noise = random."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    n = len(y)
    groups = np.zeros(n, dtype=int)
    # signal: similarity 1 if same class else 0 (+ tiny noise), symmetric
    signal = (y[:, None] == y[None, :]).astype(float)
    signal += rng.normal(0, 0.01, (n, n))
    signal = (signal + signal.T) / 2
    np.fill_diagonal(signal, 1.0)
    noise = rng.normal(0, 1, (n, n))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 1.0)
    return signal, noise, y, groups


def test_permutation_detects_signal_and_clears_noise():
    signal, noise, y, groups = _setup()
    sig = permutation_pvalue(signal, y, groups, n_perm=1000, seed=1)
    noi = permutation_pvalue(noise, y, groups, n_perm=1000, seed=1)
    assert sig["p_value"] <= 0.01, f"signal should be significant, got p={sig['p_value']}"
    assert sig["observed"] > noi["observed"], "signal mAP should exceed noise mAP"
    assert noi["p_value"] > 0.05, f"noise should NOT be significant, got p={noi['p_value']}"
    assert sig["n_queries"] > 0


def test_bootstrap_ci_brackets_map():
    signal, _, y, groups = _setup()
    ci = bootstrap_ci(signal, y, groups, n_boot=1000, seed=1)
    assert ci["ci_low"] <= ci["map"] <= ci["ci_high"]
    assert ci["ci_low"] > 0.5, "signal mAP CI should sit well above chance for 3 balanced classes"


def test_paired_bootstrap_prefers_signal():
    signal, noise, y, groups = _setup()
    res = paired_bootstrap_gt(signal, noise, y, groups, n_boot=1000, seed=1)
    assert res["delta"] > 0 and res["p_a_gt_b"] >= 0.95, res


def test_per_query_ap_skips_degenerate_queries():
    # a group with only one sibling, or all-same-label, contributes no queries
    y = np.array([0, 0, 1])
    groups = np.array([0, 0, 0])   # query 2 (label 1) has siblings {0,0} → all-different, rel.sum()==0 handled
    cos = np.eye(3) + 0.01
    aps, qidx = per_query_ap(cos, y, groups)
    # each query has 2 siblings; relevance is mixed only for the label-0 items? label0 sibs of item0 = {1(same),2(diff)} mixed
    assert all(0.0 <= a <= 1.0 for a in aps)


def test_nan_when_no_valid_queries():
    y = np.array([0, 1])
    groups = np.array([0, 1])       # no group has >=2 siblings
    cos = np.eye(2)
    out = permutation_pvalue(cos, y, groups, n_perm=100)
    assert out["p_value"] is None and out["n_queries"] == 0
