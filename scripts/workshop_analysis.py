"""The encoder head-to-head for the workshop paper: every featurizer, every axis, one command.

Runs on whatever corpus it is given, so it can be DEBUGGED on the 25-organism gate set and then re-run
unchanged on the 419-organism recipe-varied corpus. That is deliberate: the alternative is writing the
analysis for the first time on deadline day against data that has never been through it.

Four axes, and the third is the one the gate set structurally cannot provide:

  concept        recipe clamped (gate) or varied (workshop) -> does the feature retrieve CONCEPT?
  rank_alpha     concept clamped, rank varied               -> is the feature RANK-INVARIANT?
  recipe_CONTROL retrieve the RECIPE instead of the concept -> a semantic feature must FAIL this.
                 On the gate set this is vacuous (recipe is constant by construction, so every
                 featurizer scores chance and the number means nothing). It only becomes evidence on a
                 recipe-varied corpus, which is why the workshop corpus mints every concept under all
                 six RECIPE_POOL entries.
  heldout_family retrieval restricted to held-out families  -> does it generalize past trained families?

Scoring matches the gate harness exactly (centered cosine of the exact block Gram, grouped permutation
null, bootstrap CI) so numbers are comparable across the two corpora.

  PYTHONPATH=. python scripts/workshop_analysis.py --manifest assets/organisms/minted_gate.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ditloracle.probe.featurizers import (
    NormOnlyFeaturizer,
    OurSVDFeaturizer,
    ProductSketchFeaturizer,
    RankLeakFeaturizer,
    SpectralStatFeaturizer,
    SubspaceProjFeaturizer,
    U1LogRegFeaturizer,
    W2TFeaturizer,
    _FixedBase,
    _pad,
    build_fixed_schema,
)
from ditloracle.probe.significance import bootstrap_ci, permutation_pvalue


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni across the featurizers compared on one axis.

    Raised by all three peer reviewers: ~10 featurizers are scored on the same n=25-32 sample and the
    smallest p is reported as if it stood alone. Holm is the right correction here rather than plain
    Bonferroni (uniformly more powerful, same family-wise error guarantee) and it is a step-down
    procedure, so a featurizer only survives if every featurizer with a smaller p also survived.
    """
    items = sorted(((k, v) for k, v in pvals.items() if v is not None), key=lambda kv: kv[1])
    m = len(items)
    out, prev_rejected = {}, True
    for i, (k, pv) in enumerate(items):
        thresh = alpha / (m - i)
        rejected = prev_rejected and pv <= thresh
        prev_rejected = rejected
        out[k] = {"p_raw": pv, "holm_threshold": round(thresh, 5), "survives_holm": bool(rejected)}
    for k, v in pvals.items():
        if v is None:
            out[k] = {"p_raw": None, "holm_threshold": None, "survives_holm": False}
    return out
from ditloracle.encoding.svd_encoder import encode_module, usable_direction_mask
import scripts.poc1c_organism_gate as gate

TOP_K = 8
N_PERM = 2000


class SigmaOnlyFeaturizer(_FixedBase):
    """ABLATION: our_svd with the singular VECTORS deleted — normalized spectrum only.

    Load-bearing for the argument: it scores ABOVE full our_svd on the gate set (0.624 vs 0.479), so
    the direction components are not merely weak, they are actively harmful. Without this row a reader
    can believe the directions contribute a little; with it, deleting them is an improvement.
    """

    name = "sigma_only_ABLATION"

    @property
    def out_dim(self):
        return self.top_k * len(self.modules)

    def module_vec(self, name, lora):
        if name not in lora:
            return torch.zeros(self.top_k, dtype=torch.float64)
        B, A, alpha, r, rs = lora[name]
        enc = encode_module(B, A, alpha=alpha, r=r, use_rslora=rs)
        s = enc.sigma / enc.frob if enc.frob > 0 else enc.sigma
        return _pad(s, self.top_k).to(torch.float64)


class DirProdSketchFeaturizer(_FixedBase):
    """ABLATION: the design doc §B.5.3 prescription — per-direction rank-1 product uᵢvᵢᵀ, sketched.

    Sign-INVARIANT by construction ((-u)(-v)ᵀ = uvᵀ), so it isolates whether sign handling was the
    problem. It was not: this still scores ~0.53. That result is what moves the diagnosis from "we
    forgot sign-invariance" to "per-direction slot alignment is unrecoverable near degeneracy".
    """

    name = "dir_prod_ABLATION"

    def __init__(self, modules, dims, top_k, p=8, q=8):
        super().__init__(modules, dims, top_k)
        self.p, self.q = p, q
        self._proj = {}

    def _projectors(self, name):
        if name not in self._proj:
            do, di = self.dims[name]
            seed = int.from_bytes(name.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31)
            g = torch.Generator().manual_seed(seed)
            self._proj[name] = (torch.randn(do, self.p, generator=g, dtype=torch.float64) / (do ** 0.5),
                                torch.randn(di, self.q, generator=g, dtype=torch.float64) / (di ** 0.5))
        return self._proj[name]

    @property
    def out_dim(self):
        return len(self.modules) * self.top_k * self.p * self.q

    def module_vec(self, name, lora):
        blk = self.top_k * self.p * self.q
        if name not in lora:
            return torch.zeros(blk, dtype=torch.float64)
        B, A, alpha, r, rs = lora[name]
        enc = encode_module(B, A, alpha=alpha, r=r, use_rslora=rs)
        ro, ri = self._projectors(name)
        keep = usable_direction_mask(enc.sigma, 1e-6)
        out = []
        for i in range(self.top_k):
            if i >= enc.k or not bool(keep[i]):
                out.append(torch.zeros(self.p * self.q, dtype=torch.float64))
                continue
            lu = ro.T @ enc.U[:, i].to(torch.float64)
            rv = enc.V[:, i].to(torch.float64) @ ri
            out.append(torch.outer(lu, rv).reshape(-1))
        return torch.cat(out)


def _featurizers(modules, dims):
    return {
        "subspace_proj": SubspaceProjFeaturizer(modules, dims, TOP_K),
        "product_sketch": ProductSketchFeaturizer(modules, dims, TOP_K),
        "u1_logreg": U1LogRegFeaturizer(modules, dims, TOP_K),
        "w2t": W2TFeaturizer(modules, dims, TOP_K),
        "spectral_stat": SpectralStatFeaturizer(modules, dims, TOP_K),
        "our_svd": OurSVDFeaturizer(modules, dims, TOP_K),
        "sigma_only_ABLATION": SigmaOnlyFeaturizer(modules, dims, TOP_K),
        "dir_prod_ABLATION": DirProdSketchFeaturizer(modules, dims, TOP_K),
        "norm_only": NormOnlyFeaturizer(modules, dims, TOP_K),
        "rank_leak_CONTROL": RankLeakFeaturizer(modules, dims, TOP_K),
    }


def _recipe_key(r):
    return (r.get("rank"), tuple(sorted(r.get("target_modules") or [])))


def score_axis(recs, loras, label_of, subset, group_of=None, name=""):
    """Grouped retrieval score for every featurizer on one axis."""
    sel = [i for i, r in enumerate(recs) if subset(r)]
    if len(sel) < 4:
        return {"axis": name, "skipped": f"only {len(sel)} organisms"}
    R = [recs[i] for i in sel]
    L = [loras[i] for i in sel]
    labs = sorted({label_of(r) for r in R})
    if len(labs) < 2:
        return {"axis": name, "skipped": f"only {len(labs)} distinct label(s)"}
    y = np.array([labs.index(label_of(r)) for r in R])
    groups = (np.array([group_of(r) for r in R]) if group_of
              else np.zeros(len(R), dtype=int))
    modules = sorted({m for l in L for m in l})
    dims = build_fixed_schema(L, top_k=TOP_K)
    dims = {m: dims[m] for m in modules}
    out = {"axis": name, "n": len(R), "n_labels": len(labs), "featurizers": {}}
    for fname, fz in _featurizers(modules, dims).items():
        try:
            cos = gate._cosine(fz, L)
            perm = permutation_pvalue(cos, y, groups, n_perm=N_PERM, seed=0)
            ci = bootstrap_ci(cos, y, groups, n_boot=N_PERM, seed=0)
            out["featurizers"][fname] = {"mAP": perm.get("observed"), "p": perm.get("p_value"),
                                         "n_queries": perm.get("n_queries"),
                                         "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
        except Exception as e:
            out["featurizers"][fname] = {"error": str(e)[:140]}
    # family-wise correction across everything scored on this axis
    holm = holm_bonferroni({k: v.get("p") for k, v in out["featurizers"].items() if "error" not in v})
    for k, h in holm.items():
        out["featurizers"][k].update(h)
    out["multiple_comparison"] = "holm-bonferroni across featurizers, alpha=0.05"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/organisms/minted_gate.json")
    ap.add_argument("--out", default="results/workshop_encoder_comparison.json")
    a = ap.parse_args()

    recs, loras = gate.load_minted(a.manifest)
    print(f"loaded {len(recs)} organisms from {a.manifest}")
    if len(recs) < 4:
        raise SystemExit("too few organisms")
    fam_of = {}
    try:
        from ditloracle.mint import taxonomy
        for c in taxonomy.resolve_concepts(None):
            fam_of[c.key] = (c.family, c.split)
    except Exception:
        pass
    print("  axes:", Counter(r.get("axis", "none") for r in recs))

    axes = []
    # 1. concept — on the workshop corpus recipe VARIES, so this is the real test
    axes.append(score_axis(
        recs, loras, lambda r: r["primary_concept"],
        lambda r: r.get("axis") in ("concept", "none"),
        group_of=None, name="concept"))
    # 2. rank-invariance
    axes.append(score_axis(
        recs, loras, lambda r: r["primary_concept"],
        lambda r: r.get("axis") == "rank_alpha", group_of=None, name="rank_alpha"))
    # 3. THE CONTROL: retrieve RECIPE. A semantic feature must be near chance here.
    axes.append(score_axis(
        recs, loras, _recipe_key,
        lambda r: r.get("axis") in ("concept", "none"), group_of=None,
        name="recipe_CONTROL"))
    # 4. held-out families only
    axes.append(score_axis(
        recs, loras, lambda r: r["primary_concept"],
        lambda r: fam_of.get(r["primary_concept"], ("", ""))[1] == "test",
        group_of=None, name="heldout_family"))

    for ax in axes:
        if "skipped" in ax:
            print(f"\n[{ax['axis']}] SKIPPED — {ax['skipped']}")
            continue
        print(f"\n[{ax['axis']}] n={ax['n']} labels={ax['n_labels']}")
        rows = sorted(ax["featurizers"].items(),
                      key=lambda kv: -(kv[1].get("mAP") or 0))
        for fname, f in rows:
            if "error" in f:
                print(f"  {fname:24} ERROR {f['error'][:60]}")
            else:
                mark = "" if f.get("survives_holm") else "  (n.s. after Holm)"
                print(f"  {fname:24} mAP={f['mAP']:.4f} "
                      f"CI=[{f.get('ci_low', float('nan')):.3f},{f.get('ci_high', float('nan')):.3f}] "
                      f"p={f['p']}  nq={f['n_queries']}{mark}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"manifest": a.manifest, "n_organisms": len(recs),
                                       "axes": axes}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
