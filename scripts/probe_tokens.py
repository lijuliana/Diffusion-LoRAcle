#!/usr/bin/env python
"""Linear probe on the reader's OWN weight tokens. The positive control the sweeps lacked.

Two sweeps sat at floor and both were read as ambiguous between "LoRA weights carry no readable
concept signal" and "the reader pipeline is broken". A linear classifier on the identical token
tensors settles it without touching the interpreter:

  probe well above chance -> tokens are informative; the LLM interpreter is the bottleneck.
  probe at chance         -> the encoder/token path is the bottleneck; no interpreter tuning helps.

Split matches train_reader.py: hold out one adapter per concept, requiring >=3 per concept.
"""
import argparse, json, pathlib, random
import numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", default="data/tokens_projbank")
    ap.add_argument("--manifest", default="assets/organisms/provisional_workshop.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_tokens.json")
    a = ap.parse_args()

    man = json.loads(pathlib.Path(a.manifest).read_text())
    concept = {o["organism_id"]: o["primary_concept"] for o in man["organisms"]}

    X, y = [], []
    for p in sorted(pathlib.Path(a.token_cache).glob("*.pt")):
        oid = p.stem
        if oid not in concept:
            continue
        t = torch.load(p, map_location="cpu")
        t = t["tokens"] if isinstance(t, dict) else t
        X.append(t.flatten().to(torch.float32).numpy())
        y.append(concept[oid])
    if not X:
        raise SystemExit("no tokens matched the manifest")
    d = min(x.shape[0] for x in X)
    X = np.stack([x[:d] for x in X]); y = np.array(y)
    print(f"{len(X)} adapters | {len(set(y))} concepts | feature dim {d}")

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
    print(f"train {len(tr)} | held-out {len(te)} | chance {1/len(set(y)):.4f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    # 16 x 5120 = 81920 features against a few hundred adapters: fit PCA on TRAIN only
    # (fitting it on all of X would leak held-out structure into the projection).
    sc = StandardScaler().fit(X[tr])
    ncomp = min(len(tr) - 1, 256)
    pca = PCA(n_components=ncomp, random_state=a.seed).fit(sc.transform(X[tr]))
    Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
    print(f"PCA {d} -> {ncomp} dims ({pca.explained_variance_ratio_.sum():.1%} variance)")
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(Ztr, y[tr])
    tr_acc = clf.score(Ztr, y[tr])
    te_acc = clf.score(Zte, y[te])
    # top-5 too: exact-match on 150 classes is harsh even for a working probe
    pr = clf.predict_proba(Zte)
    top5 = float(np.mean([y[te][i] in clf.classes_[np.argsort(pr[i])[-5:]] for i in range(len(te))]))
    chance = 1 / len(set(y))
    print(f"\n  probe TRAIN acc   {tr_acc:.3f}")
    print(f"  probe HELD-OUT    {te_acc:.3f}   (chance {chance:.4f}, {te_acc/chance:.1f}x)")
    print(f"  probe held-out@5  {top5:.3f}")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(
        {"n": len(X), "n_concepts": len(set(y)), "dim": d, "train_acc": tr_acc,
         "heldout_acc": te_acc, "heldout_top5": top5, "chance": chance}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
