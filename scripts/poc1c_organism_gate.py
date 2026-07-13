"""POC-1c — the CAUSAL go/no-go on controlled organisms (design doc §B.7 POC-1c, §B.6.2).

This is the confound-free complement to the wild POC-1b gate. On the counterfactual matched sets
minted per `ditloracle/safety/mint_spec.py`, recipe/creator/spectrum are held fixed BY CONSTRUCTION,
so above-chance concept retrieval CANNOT be a recipe/creator signature — it can only be real weight
semantics. That makes this the cleanest possible test of the project's central premise, and (unlike
the wild gate) it needs no human labels.

Two tests, both scored with a permutation null (not a hard-coded margin):

  1. CONCEPT AXIS (recipe clamped, concept varied): does our_svd retrieve same-concept organisms
     above chance, when every organism shares one recipe? PASS ⇒ concept is in the weight directions.
  2. RANK-INVARIANCE AXIS (concept clamped, rank varied): does our_svd retrieve the SAME concept
     ACROSS different ranks? PASS ⇒ the reader is not just reading rank.

The referee is the same lineup as POC-1b (rank_leak / spectral / product_sketch), so we can see
whether our_svd's advantage is real and whether canonicalization beats the raw product.

Input: a directory of minted organism .safetensors + the ground-truth manifest (mint_spec plan with
`weights_path` filled in after minting). Until minting happens on the cluster, run with --synthetic to
exercise the harness on planted ground truth (proves the gate logic, not the science).

Run (after minting):
  PYTHONPATH=. python scripts/poc1c_organism_gate.py --manifest assets/organisms/minted_manifest.json
Dry-run the plumbing now:
  PYTHONPATH=. python scripts/poc1c_organism_gate.py --synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ditloracle.probe.featurizers import (
    OurSVDFeaturizer,
    ProductSketchFeaturizer,
    RankLeakFeaturizer,
    SpectralStatFeaturizer,
    build_fixed_schema,
)
from ditloracle.probe.significance import bootstrap_ci, permutation_pvalue

TOP_K = 8
SIG_P = 0.01           # permutation-p threshold for "above chance"


def _cosine(fz, loras):
    """center + cosine of the exact block Gram (same kernel space as poc1_probe)."""
    G = fz.gram(loras)
    n = G.shape[0]
    row = G.mean(axis=1, keepdims=True); col = G.mean(axis=0, keepdims=True)
    G = G - row - col + G.mean()
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return G / np.outer(d, d)


def load_minted(manifest_path: str):
    """Load minted organisms: returns (records, loras). Records are the ground-truth OrganismRecord
    dicts (must have weights_path); loras are canonical (B,A,...) factor dicts."""
    from ditloracle.formats.safetensors_io import load_canonical_factors
    plan = json.loads(Path(manifest_path).read_text())
    orgs = plan["organisms"] if isinstance(plan, dict) else plan
    recs, loras = [], []
    for o in orgs:
        wp = o.get("weights_path")
        if not wp or not Path(wp).exists():
            continue
        try:
            fac = load_canonical_factors(wp)
        except Exception:
            continue
        if fac:
            recs.append(o); loras.append(fac)
    return recs, loras


def make_synthetic(plan_path: str = "assets/organisms/mint_plan_poc1c.json", seed: int = 0):
    """Fabricate factor dicts whose DIRECTIONS encode the concept and whose SPECTRUM/rank match the
    ground-truth recipe cell — so the gate harness can be exercised before real minting. This is a
    plumbing check, NOT evidence: it only shows the gate CAN detect concept-in-directions when present.
    """
    plan = json.loads(Path(plan_path).read_text())
    recs = plan["organisms"]
    rng = np.random.default_rng(seed)
    d_out = d_in = 128
    modules = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"]
    # one fixed random direction basis per concept (shared across recipe cells → concept lives in
    # directions, not spectrum); spectrum set by the recipe cell's rank.
    concepts = sorted({r["primary_concept"] for r in recs})
    concept_dirs = {c: {m: (rng.standard_normal((d_out, 4)), rng.standard_normal((d_in, 4)))
                        for m in modules} for c in concepts}
    loras = []
    for r in recs:
        rk = int(r.get("rank") or 16)
        c = r["primary_concept"]
        lora = {}
        for m in modules:
            Ud, Vd = concept_dirs[c][m]
            # build rank-rk factors: first 4 columns = concept directions, rest = recipe-specific noise
            U = np.zeros((d_out, rk)); V = np.zeros((d_in, rk))
            k0 = min(4, rk)
            U[:, :k0] = Ud[:, :k0]; V[:, :k0] = Vd[:, :k0]
            if rk > k0:
                U[:, k0:] = rng.standard_normal((d_out, rk - k0)) * 0.1
                V[:, k0:] = rng.standard_normal((d_in, rk - k0)) * 0.1
            B = torch.tensor(U, dtype=torch.float32)
            A = torch.tensor(V.T, dtype=torch.float32)
            lora[m] = (B, A, float(r.get("alpha") or 16.0), rk, False)
        loras.append(lora)
    return recs, loras


def run_axis(recs, loras, axis: str, seed: int = 0) -> dict:
    """Score one counterfactual axis. Group = family_key (matched set); label = primary_concept.
    For axis='concept' each family is one set of distinct concepts under a clamped recipe; for
    axis='rank_alpha' each family is one concept across ranks (so retrieval must cross ranks)."""
    sel = [i for i, r in enumerate(recs) if r.get("axis") == axis]
    if len(sel) < 4:
        return {"axis": axis, "skipped": "not enough organisms for this axis"}
    R = [recs[i] for i in sel]; L = [loras[i] for i in sel]
    concepts = sorted({r["primary_concept"] for r in R})
    y = np.array([concepts.index(r["primary_concept"]) for r in R])

    if axis == "rank_alpha":
        # concept is clamped within a family; to test rank-invariance we retrieve SAME CONCEPT across
        # the whole pool (one big group), so siblings differ in rank — a rank reader cannot win.
        groups = np.zeros(len(R), dtype=int)
    else:
        groups = np.array([hash(r["family_key"]) for r in R])

    modules = sorted({m for l in L for m in l})
    dims = build_fixed_schema(L, top_k=TOP_K)
    dims = {m: dims[m] for m in modules}
    fzs = {
        "spectral_stat": SpectralStatFeaturizer(modules, dims, TOP_K),
        "product_sketch": ProductSketchFeaturizer(modules, dims, TOP_K),
        "our_svd": OurSVDFeaturizer(modules, dims, TOP_K),
        "rank_leak_CONTROL": RankLeakFeaturizer(modules, dims, TOP_K),
    }
    out = {"axis": axis, "n": len(R), "concepts": concepts, "featurizers": {}}
    for name, fz in fzs.items():
        try:
            cos = _cosine(fz, L)
            perm = permutation_pvalue(cos, y, groups, n_perm=2000, seed=seed)
            ci = bootstrap_ci(cos, y, groups, n_boot=2000, seed=seed)
            out["featurizers"][name] = {**perm, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
        except Exception as e:
            out["featurizers"][name] = {"error": str(e)[:120]}
    return out


def verdict(concept_axis: dict, rank_axis: dict) -> list[str]:
    v = []
    def sig(a, name):
        f = a.get("featurizers", {}).get(name, {})
        p = f.get("p_value")
        return f, (p is not None and p <= SIG_P)

    our_c, our_c_sig = sig(concept_axis, "our_svd")
    leak_c, leak_c_sig = sig(concept_axis, "rank_leak_CONTROL")

    if "skipped" in concept_axis:
        v.append(f"concept axis skipped: {concept_axis['skipped']}")
    else:
        if our_c_sig and not leak_c_sig:
            v.append(f"✓ CONCEPT-IN-WEIGHTS: our_svd retrieves concept under a CLAMPED recipe "
                     f"(mAP={our_c.get('observed')}, p={our_c.get('p_value')}), while the rank/recipe "
                     f"control is at chance (p={leak_c.get('p_value')}). Confound-free evidence the "
                     f"premise holds.")
        elif our_c_sig and leak_c_sig:
            v.append(f"⚠ our_svd is above chance (p={our_c.get('p_value')}) BUT so is the rank/recipe "
                     f"control (p={leak_c.get('p_value')}) — the matched set isn't as clamped as "
                     f"intended; inspect the recipe cells before trusting.")
        else:
            v.append(f"✗ our_svd NOT above chance on the clamped-recipe concept axis "
                     f"(p={our_c.get('p_value')}, n_queries={our_c.get('n_queries')}). Either concept "
                     f"is not in the directions, or the organism set is too small/easy — reassess.")

    if "skipped" in rank_axis:
        v.append(f"rank-invariance axis skipped: {rank_axis['skipped']}")
    else:
        our_r, our_r_sig = sig(rank_axis, "our_svd")
        leak_r, leak_r_sig = sig(rank_axis, "rank_leak_CONTROL")
        if our_r_sig and not leak_r_sig:
            v.append(f"✓ RANK-INVARIANT: our_svd retrieves the same concept ACROSS ranks "
                     f"(p={our_r.get('p_value')}) while the rank control cannot (p={leak_r.get('p_value')}).")
        else:
            v.append(f"✗ rank-invariance not shown (our_svd p={our_r.get('p_value')}, "
                     f"rank_leak p={leak_r.get('p_value')}).")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/organisms/minted_manifest.json",
                    help="minted organism manifest (mint_spec plan with weights_path filled in)")
    ap.add_argument("--synthetic", action="store_true",
                    help="fabricate concept-in-directions organisms to exercise the gate plumbing (NOT science)")
    ap.add_argument("--out", default="results/poc1c_organism_gate.json")
    args = ap.parse_args()

    if args.synthetic:
        recs, loras = make_synthetic()
        source = "SYNTHETIC (plumbing check — not evidence)"
    elif Path(args.manifest).exists():
        recs, loras = load_minted(args.manifest)
        source = args.manifest
    else:
        print(f"[abort] no minted manifest at {args.manifest}. Mint on the cluster first "
              f"(see ditloracle/safety/mint_spec.py) or run --synthetic to test plumbing.")
        return

    print(f"=== POC-1c organism gate ===\nsource: {source}\norganisms loaded: {len(recs)}")
    if len(recs) < 4:
        print("[abort] fewer than 4 organisms — mint the plan first.")
        return
    concept_axis = run_axis(recs, loras, "concept")
    rank_axis = run_axis(recs, loras, "rank_alpha")
    v = verdict(concept_axis, rank_axis)

    for ax in (concept_axis, rank_axis):
        if "skipped" in ax:
            continue
        print(f"\n[{ax['axis']} axis]  n={ax['n']}  concepts={len(ax['concepts'])}")
        for name, f in ax["featurizers"].items():
            if "error" in f:
                print(f"  {name:20} ERROR: {f['error']}")
            else:
                print(f"  {name:20} mAP={f.get('observed')}  p={f.get('p_value')}  "
                      f"CI=[{f.get('ci_low')},{f.get('ci_high')}]  nq={f.get('n_queries')}")
    print("\nVerdict:")
    for line in v:
        print("  " + line)

    result = {"source": source, "n_organisms": len(recs),
              "concept_axis": concept_axis, "rank_axis": rank_axis, "verdict": v}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
