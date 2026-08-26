"""Tests for the malicious-vs-benign detection head (`ditloracle/probe/detection.py`).

This is how the competing DETECTORS are scored (`2607.25750` = u₁ + logistic regression,
`2602.15195` = spectral stats + linear head), so the harness has to be trustworthy before any Fig-3/4
number is quoted: AUROC must be 1.0 when the planted signal is in the feature, ~chance when it is not,
out-of-fold (never in-sample), honest about falling back to a non-grouped split, and it must refuse
rather than OOM on a featurizer too wide to materialize. CPU, no downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ditloracle.probe.detection import (
    detect_from_featurizer,
    detection_auroc,
    feature_matrix,
    roc_auc_permutation_p,
    transfer_auroc,
)
from ditloracle.probe.featurizers import U1LogRegFeaturizer, build_fixed_schema

DT = torch.float64
MOD = "blk0.attn.qkv"
D, R = 24, 4
TOPK = 8


def _lora_with_top_direction(top, rng, noise=0.06, alpha=16.0):
    """A LoRA whose ΔW has `top` (+noise) as its leading left singular direction."""
    first = top + noise * torch.tensor(rng.standard_normal(D), dtype=DT)
    rest = torch.tensor(rng.standard_normal((D, R - 1)), dtype=DT)
    U, _ = torch.linalg.qr(torch.cat([first.reshape(D, 1), rest], dim=1))
    V, _ = torch.linalg.qr(torch.tensor(rng.standard_normal((D, R)), dtype=DT))
    s = torch.tensor([4.0, 2.0, 1.0, 0.5], dtype=DT)
    B = U @ torch.diag(s) * (R / alpha)
    return {MOD: (B, V.transpose(0, 1), alpha, R, False)}


def _planted_direction(rng):
    """A malicious 'signature' direction with an UNAMBIGUOUS largest entry.

    Not cosmetic: the default u₁ sign rule pivots on the largest-|entry| coordinate, so if the top two
    entries near-tie, noise moves the pivot and flips whole feature blocks. With a merely-largish
    coordinate (3σ) that flips 5–20% of adapters and swings AUROC over 0.31–1.0 across seeds; at 6σ
    the pivot is pinned and both sign rules give ≥0.99. That fragility is a property of the baseline
    (pinned in tests/test_featurizers.py::test_u1_sign_pivot_is_fragile_when_top_entries_tie); here we
    condition it away so this file tests the SCORER, not that failure mode."""
    v = torch.tensor(rng.standard_normal(D), dtype=DT)
    v[0] = 6.0
    return v / torch.linalg.vector_norm(v)


def _dataset(n_per_class=20, seed=0, planted=True):
    """Malicious adapters share ONE planted top-left direction; benign ones are random.

    `planted=False` gives the negative control: both classes are drawn identically, so any AUROC
    materially above 0.5 would mean the harness is leaking (e.g. scoring in-sample)."""
    rng = np.random.default_rng(seed)
    mal_dir = _planted_direction(rng)
    loras, y, groups = [], [], []
    for i in range(n_per_class):
        base = mal_dir if planted else torch.tensor(rng.standard_normal(D), dtype=DT)
        loras.append(_lora_with_top_direction(base, rng))
        y.append(1)
        groups.append(f"mal{i % 5}")
    for i in range(n_per_class):
        loras.append(_lora_with_top_direction(torch.tensor(rng.standard_normal(D), dtype=DT), rng))
        y.append(0)
        groups.append(f"ben{i % 5}")
    return loras, np.array(y), groups


def _u1(loras):
    dims = build_fixed_schema(loras, top_k=TOPK)
    return U1LogRegFeaturizer(sorted(dims), dims, TOPK)


def test_u1_logreg_detects_a_planted_top_direction():
    """End-to-end, exactly as the paper runs it: u₁ per module -> logistic regression -> AUROC.

    n_per_class is 100, not the fixture default of 20, and that is load-bearing rather than
    belt-and-braces. The feature is the full 24-dim u₁, scored under grouped 5-fold CV, so at n=20 each
    fold fits 24 features on 32 samples and tests on 8 — AUROC then measures seed noise, not the
    detector: across three seeds it ranged 0.305–1.000 (mean 0.68), and the default seed landed at
    0.745, failing a >0.9 assertion on a scorer that is in fact correct. Sweeping n shows the expected
    monotone convergence (mean AUROC 0.68 → 0.89 → 0.95 → 0.96 at n = 20/50/100/200) while the
    unplanted control stays at chance, which is what identifies the small-n result as variance rather
    than a leak. 100 puts the default seed at ~0.97 with room under the threshold.
    """
    loras, y, groups = _dataset(n_per_class=100)
    res = detect_from_featurizer(_u1(loras), loras, y, groups=groups, n_perm=200)
    assert res["featurizer"] == "u1_logreg"
    assert res["out_dim"] == D
    assert res["auroc"] > 0.9, res
    assert res["permutation"]["p_value"] <= 0.01, res["permutation"]
    assert res["n_pos"] == res["n_neg"] == 100
    assert len(res["roc"]["fpr"]) == len(res["roc"]["tpr"])
    assert res["grouped"] is True and "warning" not in res


def test_scores_are_out_of_fold_so_no_signal_means_chance():
    """The negative control that catches in-sample scoring: with nothing planted, a 24-dim feature is
    easily fit PERFECTLY in-sample, so a leaky harness would report AUROC ≈ 1.

    Averaged over seeds, because a single run's AUROC has SD ≈ 0.06 even under an exact null — a
    one-draw assertion here would be a flaky test, not a control."""
    aurocs = []
    for seed in range(6):
        loras, y, groups = _dataset(n_per_class=40, planted=False, seed=seed)
        res = detect_from_featurizer(_u1(loras), loras, y, groups=groups, n_perm=0)
        aurocs.append(res["auroc"])
        assert res["auroc"] < 0.8, f"seed {seed}: no signal planted but AUROC={res['auroc']} — leak"
    assert 0.35 < float(np.mean(aurocs)) < 0.65, f"mean AUROC {np.mean(aurocs)} is not chance-like"


def test_ungrouped_fallback_is_flagged_not_silent():
    """Too few groups to hold one out => plain stratified CV, which is NOT a clean generalization
    estimate. It must be labelled as such (the same honesty rule as poc1_probe.grouped_cv_kernel)."""
    loras, y, _ = _dataset()
    res = detect_from_featurizer(_u1(loras), loras, y, groups=["a"] * 20 + ["b"] * 20, n_perm=0)
    assert res["grouped"] is False
    assert "warning" in res and "not a clean generalization" in res["warning"].lower()
    # no groups at all behaves the same way
    assert detect_from_featurizer(_u1(loras), loras, y, n_perm=0)["grouped"] is False


def test_labels_must_be_binary_and_populated():
    X = np.random.default_rng(0).standard_normal((20, 5))
    with pytest.raises(ValueError):
        detection_auroc(X, np.arange(20) % 3)          # multi-class is not a detection task
    res = detection_auroc(X, np.array([1] + [0] * 19))  # a single positive -> undefined, not a crash
    assert res["auroc"] is None and "need >=2 per class" in res["note"]


def test_feature_matrix_refuses_to_materialize_an_oversized_matrix():
    """`OurSVDFeaturizer` on a wide schema is ~2.9M dims; stacking it OOM-crashed a machine once. The
    detection head has no precomputed-kernel option, so it must refuse loudly, not swap to death."""
    loras, _, _ = _dataset(n_per_class=3)
    fz = _u1(loras)
    assert feature_matrix(fz, loras).shape == (6, D)
    with pytest.raises(MemoryError, match="max_gib"):
        feature_matrix(fz, loras, max_gib=1e-9)


def test_transfer_auroc_is_the_cross_base_arm():
    """The paper's headline generalization claim: fit on one population, score another. Requires a
    SHARED feature layout — a mismatch must fail loudly rather than compare nonsense columns."""
    tr_loras, y_tr, _ = _dataset(seed=1)
    te_loras, y_te, _ = _dataset(seed=2)
    # one schema over BOTH corpora, so train and test land in the same feature space
    dims = build_fixed_schema(tr_loras + te_loras, top_k=TOPK)
    fz = U1LogRegFeaturizer(sorted(dims), dims, TOPK)
    X_tr = feature_matrix(fz, tr_loras)
    X_te = feature_matrix(fz, te_loras)
    res = transfer_auroc(X_tr, y_tr, X_te, y_te, n_perm=0)
    # different seeds => a DIFFERENT planted direction, so transfer is expected to be poor; the test
    # asserts the machinery, and that it is not silently reporting the in-domain number.
    assert res["auroc"] is not None and res["n_train"] == 40 and res["n_test"] == 40
    same = transfer_auroc(X_tr, y_tr, X_tr, y_tr, n_perm=0)     # sanity: it can fit its own domain
    assert same["auroc"] > 0.9
    with pytest.raises(ValueError, match="fixed schema"):
        transfer_auroc(X_tr, y_tr, X_te[:, :3], y_te)


def test_permutation_p_is_the_rank_null():
    y = np.array([1] * 10 + [0] * 10)
    perfect = roc_auc_permutation_p(y, np.r_[np.ones(10), np.zeros(10)], n_perm=200)
    assert perfect["auroc"] == 1.0 and perfect["p_value"] <= 0.01
    rng = np.random.default_rng(0)
    noise = roc_auc_permutation_p(y, rng.standard_normal(20), n_perm=200)
    assert noise["p_value"] > 0.01 and 0.3 < noise["null_mean"] < 0.7
