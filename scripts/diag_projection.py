#!/usr/bin/env python
"""Localise the projection failure to before, inside, or after ProjectionBank.

Three checks, none of which need the source author:

  COVERAGE  project() returns None on a shape mismatch and the caller skips that module, so an
            orientation or fused-module error shows up as silently dropped modules rather than as
            wrong numbers. Counts what actually survives.
  COLLAPSE  pairwise cosine of the injected tokens across different adapters. A working map gives a
            spread; a misport collapses everything to nearly one vector or to near-zero norms.
  SCALE     norm of the projected vector against the raw direction, to see what the map does to it.

The fourth check, whether the projection beats a random matrix of the same shape, is answered by
extracting a random_orth cache and probing it, since random_orth is exactly that control.
"""
import argparse, json, pathlib, collections
import numpy as np
import torch

from ditloracle.formats.safetensors_io import load_canonical_factors
from ditloracle.encoding.svd_encoder import compact_svd_from_factors
from ditloracle.reader.dataset import residual_side
from ditloracle.reader.projection_bank import KleinProjectionBank, writes_to_residual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/organisms/provisional_workshop.json")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--n-adapters", type=int, default=20)
    ap.add_argument("--out", default="results/diag_projection.json")
    a = ap.parse_args()

    import os, glob
    base = a.base_dir or os.environ.get("KLEIN_BASE_DIR") or next(
        iter(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein*/snapshots/*"))), None)
    if base is None:
        raise SystemExit("need the klein base checkpoint; set KLEIN_BASE_DIR")
    bank = KleinProjectionBank.from_safetensors(base)

    man = json.loads(pathlib.Path(a.manifest).read_text())
    orgs = (man["organisms"] if isinstance(man, dict) else man)[:a.n_adapters]

    kept, dropped = collections.Counter(), collections.Counter()
    vecs, ratios = [], []
    for rec in orgs:
        wp = rec.get("weights_path")
        if not wp or not pathlib.Path(wp).exists():
            continue
        try:
            fac = load_canonical_factors(wp)
        except Exception:
            continue
        per_adapter = []
        for name in sorted(fac):
            B, A, alpha, r, rs = fac[name]
            scale = 1.0 if alpha is None else (alpha / (r ** 0.5) if rs else alpha / r)
            try:
                U, S, V = compact_svd_from_factors(B, A, scale=scale)
            except Exception:
                continue
            if S.numel() == 0:
                continue
            M = U if residual_side(name) == "U" else V
            d = M[:, 0].float()
            pv = bank.project(name, d.cpu())
            if pv is None:
                dropped[f"{'resid' if writes_to_residual(name) else 'mapped'}:{tuple(d.shape)}"] += 1
            else:
                kept[f"{'resid' if writes_to_residual(name) else 'mapped'}"] += 1
                ratios.append(float(pv.norm() / (d.norm() + 1e-12)))
                per_adapter.append(pv.numpy())
        if per_adapter:
            vecs.append(np.concatenate(per_adapter))

    n_kept, n_drop = sum(kept.values()), sum(dropped.values())
    print(f"\n=== COVERAGE over {len(vecs)} adapters ===")
    print(f"  modules kept    : {n_kept}")
    print(f"  modules DROPPED : {n_drop}  ({n_drop / max(n_kept + n_drop, 1):.1%} of all modules)")
    for k, v in kept.most_common():
        print(f"    kept    {k}: {v}")
    for k, v in dropped.most_common(8):
        print(f"    dropped {k}: {v}")

    print(f"\n=== SCALE ===")
    if ratios:
        r = np.array(ratios)
        print(f"  ‖W d‖ / ‖d‖ : median {np.median(r):.4f}  min {r.min():.4f}  max {r.max():.4f}")

    print(f"\n=== COLLAPSE (pairwise cosine of injected tokens across adapters) ===")
    if len(vecs) >= 2:
        L = min(v.shape[0] for v in vecs)
        X = np.stack([v[:L] for v in vecs])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        C = Xn @ Xn.T
        iu = np.triu_indices(len(X), k=1)
        c = C[iu]
        print(f"  n={len(X)}  cosine: median {np.median(c):.4f}  mean {c.mean():.4f}  "
              f"min {c.min():.4f}  max {c.max():.4f}")
        verdict = ("COLLAPSED (all adapters nearly identical)" if np.median(c) > 0.99
                   else "healthy spread" if np.median(c) < 0.9 else "suspiciously high")
        print(f"  verdict: {verdict}")
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps({
            "n_adapters": len(X), "modules_kept": n_kept, "modules_dropped": n_drop,
            "dropped_detail": dict(dropped), "kept_detail": dict(kept),
            "cos_median": float(np.median(c)), "cos_min": float(c.min()), "cos_max": float(c.max()),
            "norm_ratio_median": float(np.median(ratios)) if ratios else None,
        }, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
