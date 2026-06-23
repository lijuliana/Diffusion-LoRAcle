"""POC-0b/c sanity: prove on SYNTHETIC ground truth that the featurizer+probe pipeline works, and
— more importantly — reproduce the central design prediction in a controlled setting:

  When synthetic 'concepts' are defined by DIRECTIONS in weight space but share the SAME spectrum,
  the spectral-stat baseline is at chance, while our SVD-direction encoding separates them.

This is the synthetic analogue of POC-1's claim and a guard that we will be measuring the right
thing on real data. CPU, no downloads.

Run: PYTHONPATH=. python3 scripts/poc0bc_baseline_sanity.py
"""

from __future__ import annotations

import json

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ditloracle.probe.featurizers import (
    OurSVDFeaturizer,
    RawABFeaturizer,
    SpectralStatFeaturizer,
    W2TFeaturizer,
    build_fixed_schema,
)

torch.manual_seed(0)
DT = torch.float64
D_OUT, D_IN, R = 48, 48, 8
MODULE = "blk0.attn.to_q"


def make_lora_with_directions(class_dirs_U, class_dirs_V, spectrum, noise=0.05, gauge=False):
    """Build a LoRA whose dominant singular directions are the class prototype (+ noise), with a
    FIXED shared spectrum across classes. So spectral stats are (in expectation) class-independent;
    only the directions carry the label.

    If gauge=True, apply a random GL(r) transform (B,A)->(B G^-1, G A): this leaves ΔW (and our
    canonical encoding) IDENTICAL, but scrambles the raw (B,A) factors — exactly the nuisance real
    adapters carry from random LoRA init. Demonstrates why a gauge-variant featurizer fails."""
    r = spectrum.shape[0]
    U = class_dirs_U + noise * torch.randn(D_OUT, r, dtype=DT)
    V = class_dirs_V + noise * torch.randn(D_IN, r, dtype=DT)
    U, _ = torch.linalg.qr(U)
    V, _ = torch.linalg.qr(V)
    root = torch.diag(spectrum.sqrt())
    B = U @ root
    A = root @ V.transpose(0, 1)
    if gauge:
        G = torch.randn(r, r, dtype=DT)
        while torch.linalg.matrix_rank(G) < r:
            G = torch.randn(r, r, dtype=DT)
        B = B @ torch.linalg.inv(G)
        A = G @ A
    return {MODULE: (B, A, 16.0, r, False)}


def build_dataset(n_per_class=60, n_classes=4, gauge=False):
    spectrum = torch.linspace(3.0, 0.8, R, dtype=DT)          # SHARED across all classes
    # one fixed prototype direction-set per class
    protos = []
    for _ in range(n_classes):
        Up, _ = torch.linalg.qr(torch.randn(D_OUT, R, dtype=DT))
        Vp, _ = torch.linalg.qr(torch.randn(D_IN, R, dtype=DT))
        protos.append((Up, Vp))
    loras, labels = [], []
    for c in range(n_classes):
        Up, Vp = protos[c]
        for _ in range(n_per_class):
            loras.append(make_lora_with_directions(Up, Vp, spectrum, gauge=gauge))
            labels.append(c)
    return loras, torch.tensor(labels), spectrum


def probe_accuracy(featurizer, loras, labels):
    X = torch.stack([featurizer.features(l) for l in loras]).numpy()
    y = labels.numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0)
    scores = cross_val_score(clf, X, y, cv=5)
    return float(scores.mean()), float(scores.std())


def main():
    mods = [MODULE]
    results = {"no_gauge": {}, "with_gauge": {}}
    # Two conditions: clean factors, and per-instance GL(r)-randomized factors (the realistic nuisance).
    for cond, gauge in [("no_gauge", False), ("with_gauge", True)]:
        loras, labels, _ = build_dataset(gauge=gauge)
        dims = build_fixed_schema(loras, top_k=R)
        for fz in [SpectralStatFeaturizer(mods, dims, R), RawABFeaturizer(mods, dims, R),
                   W2TFeaturizer(mods, dims, R), OurSVDFeaturizer(mods, dims, R)]:
            acc, std = probe_accuracy(fz, loras, labels)
            results[cond][fz.name] = {"cv_acc_mean": round(acc, 4), "cv_acc_std": round(std, 4)}
    results["chance"] = 0.25
    results["setup"] = (
        "4 classes defined by SVD directions, SHARED spectrum across classes. "
        "no_gauge: clean factors. with_gauge: per-instance GL(r) randomization of (B,A) — leaves ΔW "
        "and our canonical encoding invariant but scrambles raw A/B (the realistic LoRA-init nuisance)."
    )
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
