#!/usr/bin/env python
"""The control the paper calls missing: how well does a linear model do on the interpreter's OWN split?

train_reader.py's adapter-level split is reproduced exactly by reading the organism ids out of a
finished run's result JSON, rather than re-deriving the split here. Re-deriving it would silently
drift the moment either side's shuffle or threshold changed, and the whole point of this control is
that it faces the same 429 training adapters and the same 96 held-out adapters as the interpreter.

Two feature families are probed:
  tokens_meanpool  the mean-pooled interpreter token, i.e. exactly what the interpreter is fed
  weight features  the encoders of Appendix B, refit on this split instead of the earlier snapshot

If these land near the interpreter's 1/384, nothing reads this corpus at this size. If they land
well above it, the representation carries signal the interpreter fails to use.
"""
import argparse, json, pathlib, sys
import numpy as np


def split_from_run(path):
    R = json.loads(pathlib.Path(path).read_text())["results"]
    tr = {s["organism"] for s in R["train"]["samples"]}
    te = {s["organism"] for s in R["heldout_adapter"]["samples"]}
    assert not (tr & te), f"split is not clean: {len(tr & te)} shared adapters"
    return tr, te


def fit_report(name, X, y, tr_i, te_i, n_classes, seed, out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    sc = StandardScaler().fit(X[tr_i])
    # cap by feature count too: some encoders emit fewer than 256 dims and PCA errors out
    k = min(len(tr_i) - 1, X.shape[1], 256)
    pca = PCA(n_components=k, random_state=seed).fit(sc.transform(X[tr_i]))
    Ztr, Zte = pca.transform(sc.transform(X[tr_i])), pca.transform(sc.transform(X[te_i]))
    clf = LogisticRegression(max_iter=5000).fit(Ztr, y[tr_i])
    tr_a, te_a = clf.score(Ztr, y[tr_i]), clf.score(Zte, y[te_i])
    pr = clf.predict_proba(Zte)
    t5 = float(np.mean([y[te_i][i] in clf.classes_[np.argsort(pr[i])[-5:]] for i in range(len(te_i))]))
    # nearest-neighbour over the same standardized features, for a non-parametric reference
    A = sc.transform(X[tr_i]); B = sc.transform(X[te_i])
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    nn = float(np.mean(y[tr_i][np.argmax(B @ A.T, axis=1)] == y[te_i]))
    chance = 1 / n_classes
    print(f"{name:<22}{tr_a:>8.3f}{te_a:>10.3f}{te_a/chance:>9.1f}{t5:>8.3f}{nn:>9.3f}"
          f"{f'{round(te_a*len(te_i))}/{len(te_i)}':>10}")
    out[name] = {"train": tr_a, "heldout": te_a, "x_chance": te_a / chance,
                 "top5": t5, "nearest_neighbour": nn, "n_heldout": len(te_i)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/sweep8/e25_real.json")
    ap.add_argument("--manifest", default="assets/organisms/provisional_workshop.json")
    ap.add_argument("--token-cache", default="data/tokens_psketch_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_reader_split.json")
    ap.add_argument("--skip-weights", action="store_true")
    a = ap.parse_args()

    tr_ids, te_ids = split_from_run(a.run)
    print(f"split from {a.run}: train {len(tr_ids)} adapters | held-out {len(te_ids)} adapters")

    plan = json.loads(pathlib.Path(a.manifest).read_text())
    orgs = plan["organisms"] if isinstance(plan, dict) else plan
    concept = {o["organism_id"]: o["primary_concept"] for o in orgs if o.get("primary_concept")}

    out, cache = {}, pathlib.Path(a.token_cache)

    # ---- the interpreter's own input -------------------------------------------------------
    import torch
    ids, feats = [], []
    for oid in sorted(tr_ids | te_ids):
        f = cache / f"{oid}.pt"
        if not f.exists() or oid not in concept:
            continue
        t = torch.load(f, map_location="cpu")["tokens"].float()
        ids.append(oid); feats.append(torch.cat([t.mean(0), t.std(0)]).numpy())
    X = np.stack(feats).astype(np.float64)
    y = np.array([concept[i] for i in ids])
    tr_i = np.array([k for k, i in enumerate(ids) if i in tr_ids])
    te_i = np.array([k for k, i in enumerate(ids) if i in te_ids])
    n_cls = len(set(y[tr_i]))
    print(f"tokens found for {len(ids)} adapters | train {len(tr_i)} | held-out {len(te_i)} | "
          f"{n_cls} training concepts, chance {1/n_cls:.4f}\n")
    print(f"{'representation':<22}{'train':>8}{'held-out':>10}{'xchance':>9}{'top5':>8}{'NN':>9}{'k/n':>10}")
    fit_report("tokens_meanpool_std", X, y, tr_i, te_i, n_cls, a.seed, out)

    # ---- the Appendix B encoders, refit on this split ---------------------------------------
    if not a.skip_weights:
        from ditloracle.probe.featurizers import (
            SubspaceProjFeaturizer, ProductSketchFeaturizer, U1LogRegFeaturizer,
            OurSVDFeaturizer, RankLeakFeaturizer, build_fixed_schema)
        import scripts.poc1c_organism_gate as gate
        recs, L = gate.load_minted(a.manifest)
        keep = [i for i, r in enumerate(recs) if r["organism_id"] in (tr_ids | te_ids)]
        recs = [recs[i] for i in keep]; L = [L[i] for i in keep]
        wid = [r["organism_id"] for r in recs]
        yw = np.array([r["primary_concept"] for r in recs])
        twi = np.array([k for k, i in enumerate(wid) if i in tr_ids])
        tei = np.array([k for k, i in enumerate(wid) if i in te_ids])
        modules = sorted({m for lora in L for m in lora})
        dims = build_fixed_schema(L, top_k=8); dims = {m: dims[m] for m in modules}
        for name, fz in (("product_sketch", ProductSketchFeaturizer(modules, dims, 8)),
                         ("u1_logreg", U1LogRegFeaturizer(modules, dims, 8)),
                         ("subspace_proj", SubspaceProjFeaturizer(modules, dims, 8)),
                         ("our_svd", OurSVDFeaturizer(modules, dims, 8)),
                         ("rank_leak_CONTROL", RankLeakFeaturizer(modules, dims, 8))):
            Xw = np.stack([fz.features(l).numpy() for l in L]).astype(np.float64)
            fit_report(name, Xw, yw, twi, tei, n_cls, a.seed, out)

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(
        {"run": a.run, "n_train": len(tr_ids), "n_heldout": len(te_ids),
         "n_classes": n_cls, "results": out}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
