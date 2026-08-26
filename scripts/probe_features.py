#!/usr/bin/env python
"""Does ANY representation carry concept signal at the scale the reader actually operates at?

The gate that authorised the reader retrieves over a CLAMPED-RECIPE subset: 32 organisms, 8
concepts, and it stays that size even when handed a 625-adapter manifest. The reader faces 150
concepts. Retrieval mAP at 8-way and 150-way classification are not the same claim, and the second
was never tested.

This runs one classifier over several representations on the SAME split as train_reader.py
(one held-out adapter per concept), so encoder quality and task scale can be separated:

  subspace_proj strong here  -> signal exists at 150-way; the reader is fed the wrong tokens.
  subspace_proj at chance    -> the 8-way gate never licensed the 150-way claim.
"""
import argparse, json, pathlib, random
import numpy as np

from ditloracle.probe.featurizers import (
    SubspaceProjFeaturizer, ProductSketchFeaturizer, U1LogRegFeaturizer,
    OurSVDFeaturizer, RankLeakFeaturizer, build_fixed_schema,
)
import scripts.poc1c_organism_gate as gate

TOP_K = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/organisms/provisional_workshop.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_features.json")
    a = ap.parse_args()

    recs, L = gate.load_minted(a.manifest)
    y = np.array([r["primary_concept"] for r in recs])
    print(f"{len(L)} adapters | {len(set(y))} concepts")

    modules = sorted({m for lora in L for m in lora})
    dims = build_fixed_schema(L, top_k=TOP_K)
    dims = {m: dims[m] for m in modules}
    fzs = {
        "subspace_proj": SubspaceProjFeaturizer(modules, dims, TOP_K),
        "product_sketch": ProductSketchFeaturizer(modules, dims, TOP_K),
        "u1_logreg": U1LogRegFeaturizer(modules, dims, TOP_K),
        "our_svd": OurSVDFeaturizer(modules, dims, TOP_K),
        "rank_leak_CONTROL": RankLeakFeaturizer(modules, dims, TOP_K),
    }

    rng = random.Random(a.seed)
    by = {}
    for i, c in enumerate(y):
        by.setdefault(c, []).append(i)
    tr, te = [], []
    for c, idx in by.items():
        rng.shuffle(idx)
        if len(idx) >= 3:
            te.append(idx[0]); tr.extend(idx[1:])
        else:
            tr.extend(idx)
    chance = 1 / len(set(y))
    print(f"train {len(tr)} | held-out {len(te)} | chance {chance:.4f}\n")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    res = {}
    print(f"{'representation':<20}{'train':>8}{'held-out':>10}{'xchance':>9}{'top5':>7}")
    for name, fz in fzs.items():
        X = np.stack([fz.features(lora).numpy() for lora in L]).astype(np.float64)
        sc = StandardScaler().fit(X[tr])
        n = min(len(tr) - 1, 256)
        pca = PCA(n_components=n, random_state=a.seed).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        clf = LogisticRegression(max_iter=5000).fit(Ztr, y[tr])
        tr_a, te_a = clf.score(Ztr, y[tr]), clf.score(Zte, y[te])
        pr = clf.predict_proba(Zte)
        t5 = float(np.mean([y[te][i] in clf.classes_[np.argsort(pr[i])[-5:]] for i in range(len(te))]))
        res[name] = {"train": tr_a, "heldout": te_a, "x_chance": te_a / chance, "top5": t5}
        print(f"{name:<20}{tr_a:>8.3f}{te_a:>10.3f}{te_a / chance:>9.1f}{t5:>7.3f}")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(
        {"n": len(L), "n_concepts": len(set(y)), "chance": chance, "results": res}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
