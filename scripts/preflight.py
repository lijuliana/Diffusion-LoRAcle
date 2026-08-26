#!/usr/bin/env python
"""Checks that must pass BEFORE any GPU job. Every one of these encodes a bug that already cost us.

Run:  PYTHONPATH=. python scripts/preflight.py --token-cache data/tokens_psketch

Each check prints PASS or FAIL with the number that decided it. A FAIL means do not launch.
The point is that a failed run should be impossible to start, not merely diagnosable afterwards.
"""
import argparse, json, pathlib, random, sys, collections
import numpy as np
import torch

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILED.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--manifest", default="assets/organisms/provisional_workshop.json")
    ap.add_argument("--d-model", type=int, default=5120)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    man = json.loads(pathlib.Path(a.manifest).read_text())
    orgs = man["organisms"] if isinstance(man, dict) else man
    concept = {o["organism_id"]: o["primary_concept"] for o in orgs}
    rank = {o["organism_id"]: str(o.get("rank")) for o in orgs}
    cd = pathlib.Path(a.token_cache)
    files = sorted(cd.glob("*.pt"))

    print(f"\n=== preflight: {a.token_cache} ===")
    check("cache non-empty", len(files) > 0, f"{len(files)} token files")
    if not files:
        sys.exit(1)

    X, y, r_lab, ids, shapes, bad, n_tokens_list = [], [], [], [], set(), [], []
    for f in files:
        oid = f.stem
        if oid not in concept:
            continue
        blob = torch.load(f, map_location="cpu")
        t = blob["tokens"] if isinstance(blob, dict) else blob
        shapes.add(tuple(t.shape[1:]))
        if not torch.isfinite(t).all():
            bad.append(oid)
        n_tokens_list.append(t)
        X.append(t.flatten().to(torch.float32).numpy())
        y.append(concept[oid]); r_lab.append(rank[oid]); ids.append(oid)

    check("tokens match manifest", len(X) > 0, f"{len(X)} of {len(files)} files matched an organism")
    if not X:
        sys.exit(1)
    check("uniform token width", len(shapes) == 1, f"widths seen: {sorted(shapes)}")

    # Token COUNT may vary (product_sketch emits one token per module). That is allowed, but any
    # code pairing one adapter's tokens with another's placeholder prefix must size the prefix per
    # adapter. Reported so a variable-length cache is never a surprise to the caller.
    counts = collections.Counter(int(t_.shape[0]) for t_ in n_tokens_list)
    check("token counts recorded", True,
          ("uniform at %d" % next(iter(counts))) if len(counts) == 1
          else "VARIABLE %s — prefix length must be computed per adapter" % dict(sorted(counts.items())))
    check("all finite (no NaN/inf)", not bad, f"{len(bad)} files with non-finite values")

    # d_token must equal the reader's hidden size, or parameter-free injection is impossible.
    w = next(iter(shapes))[-1]
    check("d_token == d_model", w == a.d_model, f"token width {w}, reader width {a.d_model}")

    d = min(x.shape[0] for x in X)
    Xa = np.stack([x[:d] for x in X]); ya = np.array(y); ra = np.array(r_lab)

    # A representation of all-zeros or all-identical tokens silently trains to the label prior.
    norms = np.linalg.norm(Xa, axis=1)
    check("no zero-norm adapters", float(norms.min()) > 1e-8, f"min ‖token‖ = {norms.min():.3e}")
    uniq = len({x.tobytes() for x in Xa.astype(np.float32)})
    check("adapters are distinct", uniq == len(Xa), f"{uniq} distinct of {len(Xa)}")

    n_tok = next(iter({t for t in shapes}), None)
    per = torch.load(files[0], map_location="cpu")
    mids = per["module_ids"] if isinstance(per, dict) else None
    if mids is not None:
        n_mod_total = len(set(mids.tolist()))
        keep = mids[:a.max_tokens].tolist()
        check("token budget covers all modules",
              len(set(keep)) == n_mod_total,
              f"{len(set(keep))} of {n_mod_total} modules within max_tokens={a.max_tokens}")

    # ---- split, built exactly as train_reader.py builds it ----
    rng = random.Random(a.seed)
    by = collections.defaultdict(list)
    for i, c in enumerate(ya):
        by[c].append(i)
    tr, te = [], []
    for c, idx in by.items():
        rng.shuffle(idx)
        if len(idx) >= 3:
            te.append(idx[0]); tr.extend(idx[1:])
        else:
            tr.extend(idx)
    check("splits disjoint", not (set(tr) & set(te)), f"train {len(tr)}, held-out {len(te)}")
    check("held-out concepts appear in train",
          set(ya[te]).issubset(set(ya[tr])),
          f"{len(set(ya[te]) - set(ya[tr]))} held-out concepts absent from train")
    check("held-out large enough to detect signal", len(te) >= 30,
          f"n={len(te)}; 3 correct reaches p<0.05 at {len(set(ya))} concepts")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from math import comb

    def fit(Xm, lab):
        sc = StandardScaler().fit(Xm[tr])
        n = min(len(tr) - 1, 256)
        pca = PCA(n_components=n, random_state=a.seed).fit(sc.transform(Xm[tr]))
        Z1, Z2 = pca.transform(sc.transform(Xm[tr])), pca.transform(sc.transform(Xm[te]))
        clf = LogisticRegression(max_iter=5000).fit(Z1, lab[tr])
        return clf.score(Z2, lab[te])

    ncls = len(set(ya))
    chance = 1 / ncls
    acc = fit(Xa, ya)
    k, n = round(acc * len(te)), len(te)
    p = 1 - sum(comb(n, i) * chance**i * (1 - chance)**(n - i) for i in range(k))
    check("representation carries concept", p < 0.05,
          f"held-out {acc:.3f} ({k}/{n}), chance {chance:.4f}, p={p:.2e}")

    # Shuffling the labels must destroy it. If it does not, the pipeline is scoring something else.
    ysh = ya.copy(); rs = np.random.default_rng(a.seed); rs.shuffle(ysh)
    acc_sh = fit(Xa, ysh)
    check("label-shuffle control collapses", acc_sh <= max(3 * chance, acc / 2),
          f"shuffled-label held-out {acc_sh:.3f} vs real {acc:.3f}")

    # Recipe leakage: if rank is MORE predictable than concept, the concept number is suspect.
    if len(set(ra)) > 1:
        acc_r = fit(Xa, ra)
        ch_r = 1 / len(set(ra))
        check("concept is not just rank",
              (acc / chance) > (acc_r / ch_r),
              f"concept {acc / chance:.1f}x chance vs rank {acc_r / ch_r:.1f}x chance")

    print()
    if FAILED:
        print(f"PREFLIGHT FAILED ({len(FAILED)}): {', '.join(FAILED)}")
        print("Do not launch.")
        sys.exit(1)
    print("PREFLIGHT PASSED. Safe to launch.")


if __name__ == "__main__":
    main()
