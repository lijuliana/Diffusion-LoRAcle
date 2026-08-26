"""Does the encoder still read concept after an adapter is perturbed? (PLAN §8, table stakes.)

`2607.25750` reports an evasion sweep, so reviewers now expect one. This adopts the SHAPE of their
protocol, not their method: perturb each adapter, re-run concept retrieval, report the drop. Perturbing
after minting means no GPU is needed and nothing has to be re-trained.

Five perturbations, each chosen because a real actor could apply it and none of them changes what the
adapter depicts:
  noise_sigma   additive Gaussian noise on B and A, scaled to a fraction of each factor's own norm
  rescale       B -> cB, A -> A/c. Leaves DeltaW EXACTLY unchanged, so any feature that moves under it
                is reading the factorisation rather than the adapter. This is the GL(r) gauge in its
                simplest form and doubles as a correctness check on the invariance claim.
  fp16          round-trip through float16
  int8          uniform per-tensor quantisation to 8 bits
  int4          uniform per-tensor quantisation to 4 bits

  PYTHONPATH=. python scripts/robustness_sweep.py --manifest assets/organisms/minted_gate_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ditloracle.probe.featurizers import (
    OurSVDFeaturizer, ProductSketchFeaturizer, SpectralStatFeaturizer,
    SubspaceProjFeaturizer, U1LogRegFeaturizer, build_fixed_schema,
)
from ditloracle.probe.significance import permutation_pvalue
import scripts.poc1c_organism_gate as gate

TOP_K = 8


def _quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return x.clone()
    levels = 2 ** bits - 1
    step = (hi - lo) / levels
    return (torch.round((x - lo) / step) * step + lo).to(x.dtype)


def perturb(lora, kind: str, rng: np.random.Generator):
    out = {}
    for name, (B, A, alpha, r, rs) in lora.items():
        B2, A2 = B.clone(), A.clone()
        if kind.startswith("noise"):
            frac = float(kind.split("_")[1])
            B2 = B2 + torch.tensor(rng.standard_normal(tuple(B2.shape)), dtype=B2.dtype) * (frac * B2.norm() / (B2.numel() ** 0.5))
            A2 = A2 + torch.tensor(rng.standard_normal(tuple(A2.shape)), dtype=A2.dtype) * (frac * A2.norm() / (A2.numel() ** 0.5))
        elif kind == "rescale":
            c = float(rng.uniform(0.25, 4.0))
            B2, A2 = B2 * c, A2 / c
        elif kind == "fp16":
            B2, A2 = B2.to(torch.float16).to(B2.dtype), A2.to(torch.float16).to(A2.dtype)
        elif kind in ("int8", "int4"):
            bits = 8 if kind == "int8" else 4
            B2, A2 = _quantize(B2, bits), _quantize(A2, bits)
        out[name] = (B2, A2, alpha, r, rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/organisms/minted_gate_full.json")
    ap.add_argument("--out", default="results/robustness_sweep.json")
    a = ap.parse_args()

    recs, loras = gate.load_minted(a.manifest)
    sel = [i for i, r in enumerate(recs) if r.get("axis") == "concept"]
    R = [recs[i] for i in sel]; L = [loras[i] for i in sel]
    cs = sorted({r["primary_concept"] for r in R})
    y = np.array([cs.index(r["primary_concept"]) for r in R])
    fam = {f: i for i, f in enumerate(sorted({r["family_key"] for r in R}))}
    groups = np.array([fam[r["family_key"]] for r in R])
    mods = sorted({m for l in L for m in l})
    dims = build_fixed_schema(L, top_k=TOP_K); dims = {m: dims[m] for m in mods}
    print(f"{len(R)} organisms, {len(cs)} concepts, {len(mods)} modules\n")

    conds = ["clean", "noise_0.01", "noise_0.05", "noise_0.10", "rescale", "fp16", "int8", "int4"]
    fzs = {"subspace_proj": SubspaceProjFeaturizer, "product_sketch": ProductSketchFeaturizer,
           "u1_logreg": U1LogRegFeaturizer, "our_svd": OurSVDFeaturizer,
           "spectral_stat": SpectralStatFeaturizer}
    res = {}
    print(f"{'condition':<12}" + "".join(f"{k:>17}" for k in fzs))
    for cond in conds:
        rng = np.random.default_rng(0)
        Lp = L if cond == "clean" else [perturb(l, cond, rng) for l in L]
        row = {}
        for fname, cls in fzs.items():
            cos = gate._cosine(cls(mods, dims, TOP_K), Lp)
            p = permutation_pvalue(cos, y, groups, n_perm=1000, seed=0)
            row[fname] = {"mAP": p.get("observed"), "p": p.get("p_value")}
        res[cond] = row
        print(f"{cond:<12}" + "".join(f"{row[k]['mAP']:>17.4f}" for k in fzs))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"n": len(R), "conditions": res}, indent=2))
    print(f"\nwrote {a.out}")
    base = res["clean"]
    print("\nretention vs clean (mAP_perturbed / mAP_clean):")
    for cond in conds[1:]:
        print(f"  {cond:<12}" + "".join(
            f"{res[cond][k]['mAP']/base[k]['mAP']:>17.2f}" if base[k]['mAP'] else f"{'-':>17}" for k in fzs))


if __name__ == "__main__":
    main()
