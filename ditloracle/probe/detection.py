"""Binary malicious-vs-benign detection scoring — the downstream head the weight-only DETECTORS use.

The featurizers in `featurizers.py` are scored two ways in this project. The gates (POC-1b/1c) score
them by within-group retrieval mAP from a Gram (`significance.py`), because the gate question is "do
the weights carry concept semantics". But the two competing papers we must beat are DETECTORS, and
they report ROC/AUROC from a linear classifier on their feature:

  * `2607.25750` — u₁ + **logistic regression** (`U1LogRegFeaturizer`). This module exists mainly so
    that baseline runs exactly as the paper runs it, on day one, next to ours.
  * `2602.15195` — spectral statistics + a linear head (`SpectralStatFeaturizer`).

So this is the head-to-head harness for PLAN §8 Figs 3/4: ONE scorer, any featurizer, so the u₁
baseline, the spectral baseline and our reader's features are compared under identical CV, identical
standardization and identical regularization, and no method wins on harness differences. It mirrors
the existing pattern (`scripts/poc0bc_baseline_sanity.probe_accuracy` = sklearn LogisticRegression +
CV over `featurizer.features(...)`; `scripts/poc1_probe.grouped_cv_kernel` = grouped CV honest about
falling back to non-grouped), generalized to AUROC/ROC and lifted into the package so gate scripts,
the robustness sweep and the cross-base arm can all call it.

Everything here is CPU, no downloads. sklearn is already a hard dependency (pyproject `scikit-learn>=1.4`).
"""

from __future__ import annotations

import numpy as np


def feature_matrix(featurizer, loras: list, max_gib: float = 2.0) -> np.ndarray:
    """(n, out_dim) float64 design matrix from a featurizer, with an explicit memory guard.

    Unlike the retrieval gates, a logistic regression needs the ACTUAL features, not a Gram — sklearn
    has no precomputed-kernel logistic regression. That is affordable for the detectors (u₁ is
    n_modules·d_out ≈ 80·3072 ≈ 250k, spectral is 5·n_modules) and at organism corpus sizes, but it is
    NOT affordable for `OurSVDFeaturizer` on a wide schema (~2.9M dims — a prior run OOM-crashed the
    machine stacking exactly this). We refuse loudly instead of swapping to death; use the Gram path
    (`fz.gram(loras)` + a precomputed-kernel SVM, as in `scripts/poc1_probe.py`) for those.
    """
    if not loras:
        raise ValueError("no adapters to featurize")
    dim = int(getattr(featurizer, "out_dim", 0)) or int(featurizer.features(loras[0]).numel())
    gib = len(loras) * dim * 8 / (1024 ** 3)
    if gib > max_gib:
        raise MemoryError(
            f"{getattr(featurizer, 'name', type(featurizer).__name__)}: feature matrix would be "
            f"{len(loras)}x{dim} = {gib:.1f} GiB (> max_gib={max_gib}). Score this featurizer through "
            f"the Gram path instead, or raise max_gib deliberately."
        )
    return np.stack([featurizer.features(l).numpy() for l in loras]).astype(np.float64)


def _folds(y: np.ndarray, groups, n_splits: int, seed: int):
    """CV splitter + whether the split is genuinely GROUPED.

    Returns (folds, grouped, n_splits_used). Same honesty requirement as
    `poc1_probe.grouped_cv_kernel`: when there are too few groups (or too few positives) to hold a
    group out, we fall back to plain stratified CV and SAY SO, because in that case the same
    creator/organism family can sit in train and test and the number is not a clean generalization
    estimate. Stratification is non-negotiable here — malicious sets are small, and an unstratified
    fold with zero positives makes AUROC undefined.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    min_class = int(np.min(np.bincount(y))) if len(y) else 0
    n_splits = max(2, min(n_splits, min_class))
    if groups is not None:
        groups = np.asarray(groups)
        if len(set(groups.tolist())) >= n_splits:
            spl = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(spl.split(np.zeros(len(y)), y, groups)), True, n_splits
    spl = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(spl.split(np.zeros(len(y)), y)), False, n_splits


def _logreg(C: float, seed: int, standardize: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = LogisticRegression(max_iter=5000, C=C, random_state=seed)
    # Standardization is fitted INSIDE the pipeline, i.e. per training fold — the test fold's
    # statistics never leak into the scaler. (`_FixedBase.gram` standardizes over the whole corpus;
    # that is fine for an unsupervised Gram, not for a fitted classifier.)
    return make_pipeline(StandardScaler(), clf) if standardize else clf


def roc_auc_permutation_p(y, scores, n_perm: int = 2000, seed: int = 0) -> dict:
    """Empirical p-value for 'AUROC is above 0.5', by shuffling labels against FIXED scores.

    The scores are held out-of-fold already, so the only question left is whether the ranking carries
    label information; permuting labels against the fixed score vector is the exact rank-based null
    (equivalent to the Mann-Whitney null) and costs no refits. Same (+1)/(n+1) convention as
    `significance.permutation_pvalue`, so p-values across the project read the same way.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    scores = np.asarray(scores)
    observed = float(roc_auc_score(y, scores))
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(rng.permutation(y), scores) for _ in range(n_perm)])
    return {"auroc": round(observed, 4),
            "p_value": round((int(np.sum(null >= observed)) + 1) / (n_perm + 1), 5),
            "n_perm": n_perm,
            "null_mean": round(float(null.mean()), 4)}


def detection_auroc(X, y, groups=None, n_splits: int = 5, C: float = 1.0, seed: int = 0,
                    standardize: bool = True, n_perm: int = 2000) -> dict:
    """Cross-validated logistic-regression detection: AUROC + ROC on a malicious-vs-benign set.

    `y` is binary with 1 = malicious (the positive class). `groups` (organism family / creator) is
    held out when possible so the number is a generalization estimate, not a memorization one.

    Scores are POOLED OUT-OF-FOLD decision-function values: every adapter is scored by a model that
    never saw it, then one ROC is computed over all of them. That gives a single curve to plot (Fig 3)
    and is stable when per-fold positive counts are tiny, where per-fold AUROCs are extremely noisy —
    per-fold values are still returned so the spread is visible rather than hidden.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError(f"detection labels must be binary 0/1 (1 = malicious); got {np.unique(y)}")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < 2 or n_neg < 2:
        return {"auroc": None, "note": f"need >=2 per class; got {n_pos} malicious / {n_neg} benign",
                "n": len(y), "n_pos": n_pos, "n_neg": n_neg}

    folds, grouped, n_splits_used = _folds(y, groups, n_splits, seed)
    scores = np.full(len(y), np.nan)
    fold_auroc: list[float] = []
    for tr, te in folds:
        if len(set(y[tr].tolist())) < 2:
            continue
        model = _logreg(C, seed, standardize).fit(X[tr], y[tr])
        s = model.decision_function(X[te])
        scores[te] = s
        if len(set(y[te].tolist())) == 2:
            fold_auroc.append(float(roc_auc_score(y[te], s)))
    ok = ~np.isnan(scores)
    if ok.sum() < 4 or len(set(y[ok].tolist())) < 2:
        return {"auroc": None, "note": "cross-validation produced too few scored adapters",
                "n": len(y), "n_pos": n_pos, "n_neg": n_neg, "grouped": grouped}

    auroc = float(roc_auc_score(y[ok], scores[ok]))
    fpr, tpr, thr = roc_curve(y[ok], scores[ok])
    out = {
        "auroc": round(auroc, 4),
        "auroc_fold_mean": round(float(np.mean(fold_auroc)), 4) if fold_auroc else None,
        "auroc_fold_std": round(float(np.std(fold_auroc)), 4) if fold_auroc else None,
        "auroc_folds": [round(a, 4) for a in fold_auroc],
        "roc": {"fpr": [round(float(v), 6) for v in fpr],
                "tpr": [round(float(v), 6) for v in tpr],
                "thresholds": [round(float(v), 6) for v in thr]},
        "n": int(ok.sum()), "n_pos": n_pos, "n_neg": n_neg,
        "grouped": grouped, "n_splits": n_splits_used, "chance": 0.5,
        "scores": [round(float(v), 6) for v in scores[ok]],
    }
    if n_perm:
        out["permutation"] = roc_auc_permutation_p(y[ok], scores[ok], n_perm=n_perm, seed=seed)
    if not grouped:
        out["warning"] = ("too few groups for a grouped split — fell back to plain stratified CV, so "
                          "the same family can appear in train and test; NOT a clean generalization "
                          "estimate.")
    return out


def detect_from_featurizer(featurizer, loras: list, y, groups=None, max_gib: float = 2.0,
                           **kwargs) -> dict:
    """Convenience: featurize then `detection_auroc`. The one-liner the gate scripts call.

    Usage (the u₁ baseline exactly as `2607.25750` runs it):
        dims = build_fixed_schema(loras, top_k=TOP_K)
        fz = U1LogRegFeaturizer(sorted(dims), dims, TOP_K)
        res = detect_from_featurizer(fz, loras, y_malicious, groups=family_keys)
    """
    X = feature_matrix(featurizer, loras, max_gib=max_gib)
    res = detection_auroc(X, y, groups=groups, **kwargs)
    res["featurizer"] = getattr(featurizer, "name", type(featurizer).__name__)
    res["out_dim"] = int(getattr(featurizer, "out_dim", X.shape[1]))
    return res


def transfer_auroc(X_train, y_train, X_test, y_test, C: float = 1.0, seed: int = 0,
                   standardize: bool = True, n_perm: int = 2000) -> dict:
    """Train on one population, score another — the CROSS-BASE-MODEL generalization `2607.25750`
    reports (fit on FLUX.1-dev adapters, test on FLUX.2-klein, or train-controlled → test-wild).

    Both matrices must come from the SAME featurizer instance / fixed schema, or the columns do not
    mean the same thing; build one `build_fixed_schema` over the union of both corpora and featurize
    each with it (see `U1LogRegFeaturizer`'s CROSS-BASE note).
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"feature layouts differ ({X_train.shape[1]} vs {X_test.shape[1]}) — build one "
                         f"fixed schema over BOTH corpora before comparing across bases")
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        return {"auroc": None, "note": "need both classes present in train and in test"}
    model = _logreg(C, seed, standardize).fit(X_train, y_train)
    s = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, s)
    out = {"auroc": round(float(roc_auc_score(y_test, s)), 4),
           "n_train": len(y_train), "n_test": len(y_test),
           "n_pos_test": int(y_test.sum()), "n_neg_test": int((1 - y_test).sum()),
           "roc": {"fpr": [round(float(v), 6) for v in fpr],
                   "tpr": [round(float(v), 6) for v in tpr]},
           "chance": 0.5}
    if n_perm:
        out["permutation"] = roc_auc_permutation_p(y_test, s, n_perm=n_perm, seed=seed)
    return out
