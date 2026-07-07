"""POC-1 probe harness — the project's central go/no-go (design doc §B.7, §B.7.1).

Question: do symmetry-aware weight features discriminate concept/style/identity that metadata,
recipe, and rank CANNOT — especially within-creator / within-recipe, where signatures are useless?

Modes (auto-detected by corpus size + label availability):
  * POC-1a (apparatus debug): small n and/or weak tag-derived labels. Confirms the pipeline runs,
    features are fixed-dimension (no rank leakage), and obvious signal exists. NOT a gate.
  * POC-1b (the gate): n>=~300 with blind human labels; the gate UNIT is the derived coarse
    concept-family (A1), and the gate METRIC is within-creator / within-recipe retrieval mAP — where
    the creator/recipe signature is held constant so only real weight semantics can separate concepts.

The gate referee is the pure-nuisance leakage controls (RankLeak + norm-FREE RecipeFingerprint),
which MUST be near retrieval-chance; our_svd must beat the non-signature featurizers (spectral/raw/
w2t) AND the norm-only baseline, on BOTH the creator and recipe groupings. The norm-INCLUSIVE recipe
control and creator-only are reported as DIAGNOSTICS, never the referee (A2). Cross-creator CV is a
secondary signal.

Run:
  PYTHONPATH=. python scripts/poc1_probe.py --manifest assets/corpus/manifest_civitai_dl.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import gc

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.svm import SVC

from ditloracle.formats.safetensors_io import load_canonical_factors
from ditloracle.probe.featurizers import (
    NormOnlyFeaturizer,
    OurSVDFeaturizer,
    ProductSketchFeaturizer,
    RankLeakFeaturizer,
    RawABFeaturizer,
    SpectralStatFeaturizer,
    W2TFeaturizer,
    build_fixed_schema,
)
from ditloracle.probe.concept_family import family_of
from ditloracle.probe.labels import gate_concept, load_human_labels, weak_family_label
from ditloracle.probe.metadata_baseline import MetadataTagBaseline
from ditloracle.probe.recipe_fingerprint import RecipeFingerprint

TOP_K = 8
MIN_CLASS = 6          # drop ultra-rare classes so CV is meaningful
MAX_MODULES = 60       # cap module count for a tractable fixed schema (most-common modules)
MIN_FREE_GB = 6.0      # abort before loading if less free RAM than this (don't swap the machine to death)


def _free_ram_gb() -> float | None:
    """Best-effort free physical RAM in GB (macOS vm_stat / Linux MemAvailable). None if unknown."""
    import subprocess
    try:
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)   # kB → GB
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 4096
        pages = {}
        for line in out.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            for tag in ("Pages free", "Pages speculative", "Pages inactive", "Pages purgeable"):
                if line.startswith(tag + ":"):
                    pages[tag] = int(line.split()[-1].strip("."))
        # macOS "available" ≈ free + speculative + inactive + purgeable (inactive/purgeable pages are
        # reclaimable on demand; counting only 'free' wildly undercounts, since macOS caches aggressively).
        avail = sum(pages.get(t, 0) for t in ("Pages free", "Pages speculative", "Pages inactive", "Pages purgeable"))
        return avail * page / (1024 ** 3)
    except Exception:
        return None


def pick_modules_from_keys(paths, k=MAX_MODULES):
    """Pick the shared module schema from a CHEAP key-only scan (no tensor data loaded). Returns the
    k most-common canonical (post-fused-split) module names across the corpus — the fixed schema the
    featurizers use, so we never need to hold modules outside it in RAM."""
    from ditloracle.formats.safetensors_io import canonical_module_names
    c: Counter = Counter()
    for p in paths:
        try:
            c.update(canonical_module_names(p))
        except Exception:
            continue
    return [m for m, _ in c.most_common(k)]


def load_corpus(manifest_path: str, keep_modules: set[str] | None = None):
    """Return (records, loras) for records with a local weight file that parses.

    MEMORY-BOUNDED (this loader previously held ALL factors for ALL adapters — ~82 GB of float32 on
    the full 441-file corpus, which OOM-crashes a 48 GB machine before gram() even runs). Fix: pass
    `keep_modules` (the fixed featurizer schema) so each adapter retains only the ~60 schema modules
    and the rest are never held. A free-RAM guard aborts cleanly rather than swapping to death.

    A3 base-lineage policy: each record is tagged r['_base_class'] (flux1_dev_verified /
    flux_family_unverified / off_base_flux_merge / unknown / non_flux). The CALLER decides strata.
    """
    from ditloracle.formats.base_lineage import verify_base_lineage
    records = json.loads(Path(manifest_path).read_text())
    recs, loras = [], []
    for r in records:
        lp = r.get("local_path")
        if not lp or not Path(lp).exists():
            continue
        free = _free_ram_gb()
        if free is not None and free < MIN_FREE_GB:
            print(f"[mem-guard] only {free:.1f} GB free RAM (< {MIN_FREE_GB}); stopping load at "
                  f"{len(recs)} adapters to avoid an OOM crash. Close apps or run on the cluster.")
            break
        try:
            fac = load_canonical_factors(lp, keep_modules=keep_modules)
        except Exception:
            continue
        if not fac:
            continue
        try:
            v = verify_base_lineage(lp, declared_base=r.get("base_model"))
            r["_base_class"] = v.base_class.value
        except Exception:
            r["_base_class"] = "unknown"
        recs.append(r)
        loras.append(fac)
    return recs, loras


def derive_labels(records, human, granularity="coarse"):
    """Return (labels, source). When blind human labels exist (POC-1b), the GATE label is the
    DERIVED COARSE CONCEPT-FAMILY (A1: function×subject from verified fields) — the unit the gate is
    grouped + scored on. Else (POC-1a) fall back to the weak keyword family for apparatus debugging.
    """
    if human:
        labels = [family_of(human.get(str(r["version_id"])) or {}, granularity=granularity,
                            verified_only=True) for r in records]
        return labels, f"human_audited_blind:family_{granularity}"
    return [weak_family_label(r) for r in records], "weak_tag_family"


class _MatrixGram:
    """Adapt a matrix-style baseline (X, names from `.matrix(records)`) to the harness's Gram
    interface so recipe/metadata controls run in the SAME CV/retrieval tables as the weight
    featurizers. Standardizes columns (matching _FixedBase.gram's per-feature scaling) then forms
    the exact linear-kernel Gram. n is small (≈441), feature dim is tiny, so memory is trivial."""

    def __init__(self, baseline, records):
        self._baseline = baseline.fit(records)
        X, names = baseline.matrix(records)
        self.out_dim = len(names)
        self._X = np.asarray(X, dtype=np.float64)

    def gram(self, loras=None, standardize_blocks: bool = True):
        X = self._X
        if standardize_blocks:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True)
            sd[sd < 1e-12] = 1.0     # constant (uninformative) columns -> 0 after centering
            X = (X - mu) / sd
        return X @ X.T


def center_gram(G):
    """Double-center a Gram (feature-space mean removal, done in kernel space):
    G_c = G − 1·Ḡ_row − Ḡ_col·1 + Ḡ_all. Removes the global mean feature vector so the comparison
    isn't dominated by a shared offset — the per-SAMPLE complement to gram()'s per-FEATURE scaling."""
    n = G.shape[0]
    row = G.mean(axis=1, keepdims=True)
    col = G.mean(axis=0, keepdims=True)
    return G - row - col + G.mean()


def cosine_from_gram(G):
    """Cosine-similarity matrix from a (centered) linear-kernel Gram (no feature matrix needed)."""
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return G / np.outer(d, d)


def within_group_retrieval_map(cos, y, groups):
    """mAP for nearest-neighbour concept retrieval among items sharing a group (creator or recipe).
    Within a group the creator/recipe signature is constant, so above-chance retrieval can only come
    from real semantic content — the POC-1b gate metric. Works from the cosine matrix (kernel space),
    so the full feature matrix is never materialized."""
    y = np.asarray(y); groups = np.asarray(groups)
    aps, chances = [], []
    for i in range(len(y)):
        sib = np.where((groups == groups[i]) & (np.arange(len(y)) != i))[0]
        if len(sib) < 2:
            continue
        rel = (y[sib] == y[i]).astype(float)
        if rel.sum() == 0 or rel.sum() == len(sib):
            continue
        order = np.argsort(-cos[i, sib])
        rel_sorted = rel[order]
        precision_at = np.cumsum(rel_sorted) / (np.arange(len(rel_sorted)) + 1)
        aps.append(float((precision_at * rel_sorted).sum() / rel_sorted.sum()))
        chances.append(float(rel.mean()))
    if not aps:
        return float("nan"), float("nan"), 0
    return float(np.mean(aps)), float(np.mean(chances)), len(aps)


def grouped_cv_kernel(G, y, groups, n_splits=5):
    """Cross-creator grouped CV accuracy using a PRECOMPUTED kernel (the exact Gram), so no feature
    matrix is needed. A precomputed-kernel SVM is the dual of a linear probe — same geometry, bounded
    memory. Groups = creators → no creator in both train and test.

    Returns (accuracy, grouped): `grouped` is False when there were too few groups to run GroupKFold
    and we fell back to plain StratifiedKFold — in that case the SAME creator can appear in train and
    test, so the number is NOT a clean cross-creator estimate and the caller must flag it (this
    fallback silently contradicting the cross-creator premise was a real reviewer-facing hole).
    NOTE: this CV accuracy is a SECONDARY diagnostic; the gate is the within-group retrieval metric.
    """
    y = np.asarray(y); groups = np.asarray(groups)
    uniq = len(set(groups))
    grouped = uniq >= n_splits
    if grouped:
        folds = GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)
    else:
        nsp = max(2, min(n_splits, int(np.min(np.bincount(y)))))
        folds = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=0).split(np.zeros(len(y)), y)
    accs = []
    for tr, te in folds:
        if len(set(y[tr])) < 2:
            continue
        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(G[np.ix_(tr, tr)], y[tr])
        accs.append(clf.score(G[np.ix_(te, tr)], y[te]))
    return (float(np.mean(accs)) if accs else float("nan")), grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/corpus/manifest_civitai_dl.json")
    ap.add_argument("--out", default="results/poc1_probe.json")
    ap.add_argument("--base-stratum", choices=["verified", "verified+unverified", "all"],
                    default="verified+unverified",
                    help="A3 policy: 'verified' = headline (pristine FLUX.1-dev only); "
                         "'verified+unverified' = + the unverifiable FLUX-family stratum (robustness); "
                         "off-base merges are ALWAYS excluded from the primary gate either way.")
    args = ap.parse_args()

    # MEMORY-BOUNDED two-pass load (§ load_corpus docstring): (1) cheap key-only scan to pick the
    # fixed ~60-module schema, (2) load factors pruned to that schema so we never hold 500+ modules
    # per adapter. This is what prevents the ~82 GB full-corpus OOM.
    all_records = json.loads(Path(args.manifest).read_text())
    local_paths = [r["local_path"] for r in all_records
                   if r.get("local_path") and Path(r["local_path"]).exists()]
    schema_modules = set(pick_modules_from_keys(local_paths))
    records, loras = load_corpus(args.manifest, keep_modules=schema_modules)
    # A3 strata. Off-base merges are excluded from the PRIMARY gate (violate fixed-base symmetry,
    # §B.4.4) — parked for the H5 cross-model arm. Then select the requested verified/unverified set.
    base_counts = Counter(r.get("_base_class", "unknown") for r in records)
    def in_stratum(r):
        bc = r.get("_base_class", "unknown")
        if bc == "off_base_flux_merge":
            return False                                   # never in the primary gate
        if args.base_stratum == "verified":
            return bc == "flux1_dev_verified"
        if args.base_stratum == "verified+unverified":
            return bc in ("flux1_dev_verified", "flux_family_unverified")
        return True                                        # 'all' (still excludes off-base above)
    keep = [i for i, r in enumerate(records) if in_stratum(r)]
    records = [records[i] for i in keep]; loras = [loras[i] for i in keep]
    n = len(records)
    human = load_human_labels()
    labels, label_source = derive_labels(records, human)

    # keep only labeled, frequent-enough classes
    keep = [i for i, l in enumerate(labels) if l is not None]
    records = [records[i] for i in keep]; loras = [loras[i] for i in keep]; labels = [labels[i] for i in keep]
    cls_counts = Counter(labels)
    good = {c for c, k in cls_counts.items() if k >= MIN_CLASS}
    keep = [i for i, l in enumerate(labels) if l in good]
    records = [records[i] for i in keep]; loras = [loras[i] for i in keep]; labels = [labels[i] for i in keep]

    mode = "POC-1b (GATE)" if (human and len(records) >= 300) else "POC-1a (apparatus debug — NOT a gate)"
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    groups = [r.get("creator") or f"u{i}" for i, r in enumerate(records)]
    chance = max(Counter(labels).values()) / len(labels) if labels else float("nan")

    print(f"=== POC-1 probe ===")
    print(f"mode: {mode}")
    print(f"base strata (full corpus): {dict(base_counts)}  | gate stratum: {args.base_stratum} "
          f"(off-base merges always excluded from primary gate)")
    print(f"label source: {label_source} | n labeled = {len(records)} | classes = {dict(Counter(labels))}")
    print(f"creators: {len(set(groups))} (>1 adapter: {sum(1 for v in Counter(groups).values() if v>1)})")
    print(f"majority-class chance = {chance:.3f}")

    if len(records) < 12 or len(classes) < 2:
        print("\n[abort] not enough labeled data to probe even in apparatus-debug mode.")
        Path(args.out).write_text(json.dumps({"mode": mode, "n": len(records), "note": "insufficient"}, indent=2))
        return

    # modules = the schema we scanned pre-load, restricted to those actually present after loading
    # (a scanned name can be absent if its file failed to load / was filtered by stratum).
    present = {m for lora in loras for m in lora}
    dims = build_fixed_schema(loras, top_k=TOP_K)
    modules = [m for m in sorted(schema_modules & present)]
    dims = {m: dims[m] for m in modules}

    featurizers = {
        "spectral_stat": SpectralStatFeaturizer(modules, dims, TOP_K),
        "norm_only": NormOnlyFeaturizer(modules, dims, TOP_K),   # A2: ΔW-norm baseline (must be beaten)
        "raw_ab": RawABFeaturizer(modules, dims, TOP_K),
        "w2t_svd": W2TFeaturizer(modules, dims, TOP_K),
        # principled GL-invariant baseline (Putterman GL-net's endorsed product feature): a fixed
        # linear sketch of ΔW, invariant to GL(r)/sign WITHOUT canonicalization — so if our_svd only
        # ties this, canonicalization isn't buying us anything over the raw product.
        "product_sketch": ProductSketchFeaturizer(modules, dims, TOP_K),
        "our_svd": OurSVDFeaturizer(modules, dims, TOP_K),
        "rank_leak_CONTROL": RankLeakFeaturizer(modules, dims, TOP_K),
        # rich recipe-signature control (§B.7.2 A2): rank, α, α/r, dtype, scheme, DoRA, target-module
        # set, fused-layout, ΔW-norm distribution — all read from the weights. A STRONG recipe control
        # (vs the thin rank_leak) so "near-chance" means the split is clean, not that the control is weak.
        "recipe_fp_CONTROL": _MatrixGram(RecipeFingerprint(include_norms=True), records),
        # ablation: same fingerprint WITHOUT the ΔW-norm block (the feature that partly correlates with
        # the semantic signal we want the reader to use — see recipe_fingerprint.py caveat).
        "recipe_fp_nonorm_CONTROL": _MatrixGram(RecipeFingerprint(include_norms=False), records),
        # METADATA/TAG baseline in the SAME retrieval lineup (scientific-gap fix): the claim is "weights
        # beat metadata", so a bag-of-words-over-tags predictor must be scored head-to-head, not just
        # asserted. When the gate label is tag-derived (POC-1a) this is expected to be strong — that is
        # the point (it shows the weak-label confound); at POC-1b (human labels) it is a real competitor.
        "metadata_tags_DIAG": _MatrixGram(MetadataTagBaseline(min_df=3), records),
    }

    # recipe groups (rank band) for the within-recipe gate: hold rank constant so the recipe
    # signature cannot separate concepts — only real weight semantics can. Rank is read from the
    # weights (not the manifest, which lacks it): the modal per-module rank of each adapter.
    def adapter_rank(lora):
        rs = [t[3] for t in lora.values() if len(t) > 3 and t[3]]  # (B,A,alpha,r,...) → r
        return int(Counter(rs).most_common(1)[0][0]) if rs else 0
    recipe_groups = [f"r{adapter_rank(l)}" for l in loras]

    results = {"mode": mode, "label_source": label_source, "n": len(records),
               "base_strata_full_corpus": dict(base_counts), "gate_base_stratum": args.base_stratum,
               "classes": dict(Counter(labels)), "chance": round(chance, 4),
               "n_creators": len(set(groups)), "cross_creator_cv": {},
               "within_creator_mAP": {}, "within_recipe_mAP": {}}

    wc_chance = wr_chance = float("nan")  # retrieval chance baselines (label-only; same across featurizers)
    wc_n = wr_n = 0
    results["errors"] = {}          # featurizers that CRASHED — kept DISTINCT from "didn't beat" (a crash
                                    # must never masquerade as a clean gate outcome via silent NaN).
    cos_cache: dict[str, np.ndarray] = {}   # keep cosine matrices for the significance pass (small: n×n)
    # MEMORY DISCIPLINE (a prior run OOM-crashed the machine by stacking the ~2.9M-dim feature matrix).
    # We never form that matrix. fz.gram() accumulates the EXACT n×n linear-kernel Gram one module-block
    # at a time (peak ≈ n×max_block ≈ 441×49k×4B ≈ 86 MB, freed per block). The probe (precomputed-kernel
    # SVM) and retrieval (cosine-from-Gram) both live in this n×n space — exact, no JL approximation.
    for name, fz in featurizers.items():
        try:
            G = fz.gram(loras)                       # exact n×n, bounded memory, per-feature standardized
            G = center_gram(G)                        # per-sample double-centering (fairness control #2)
            cos = cosine_from_gram(G)
            acc, cv_grouped = grouped_cv_kernel(cos, y, groups)  # cosine kernel = scale-normalized, fair
            results["cross_creator_cv"][name] = round(acc, 4)
            results["cv_is_cross_creator"] = cv_grouped   # False ⇒ fell back to non-grouped CV (leaky)
            wc_map, wc_chance, wc_n = within_group_retrieval_map(cos, y, groups)
            wr_map, wr_chance, wr_n = within_group_retrieval_map(cos, y, recipe_groups)
            results["within_creator_mAP"][name] = round(wc_map, 4)
            results["within_recipe_mAP"][name] = round(wr_map, 4)
            cos_cache[name] = cos.copy()
            print(f"  {name:20} xcreator-CV={acc:.3f}  withinCreator-mAP={wc_map:.3f}  withinRecipe-mAP={wr_map:.3f}  (dim={fz.out_dim})")
            del G
            gc.collect()
        except Exception as e:
            # record as an EXPLICIT error, not a silent NaN in the score dicts — the verdict checks this.
            results["errors"][name] = str(e)[:160]
            print(f"  {name:20} ERROR: {str(e)[:100]}")
    results["within_creator_mAP_chance"] = round(wc_chance, 4) if wc_chance == wc_chance else None
    results["within_recipe_mAP_chance"] = round(wr_chance, 4) if wr_chance == wr_chance else None
    results["n_queries"] = {"within_creator": int(wc_n), "within_recipe": int(wr_n)}
    print(f"  [retrieval queries: within-creator={wc_n}, within-recipe={wr_n}]")

    # ---- SIGNIFICANCE (replaces hard-coded margins with permutation nulls + paired bootstraps) ----
    # For the gate we care about two questions, each answered with a real test, not a +0.05 threshold:
    #   (1) is a featurizer's within-group retrieval ABOVE CHANCE?  → permutation p-value (label shuffle)
    #   (2) does our_svd BEAT a given baseline, accounting for query noise? → paired bootstrap P(Δ>0)
    from ditloracle.probe.significance import paired_bootstrap_gt, permutation_pvalue
    sig = {"within_creator": {}, "within_recipe": {}}
    for grp_name, grp in (("within_creator", groups), ("within_recipe", recipe_groups)):
        for name, cos in cos_cache.items():
            sig[grp_name][name] = permutation_pvalue(cos, y, np.asarray(grp), n_perm=2000, seed=0)
    # our_svd vs each competing featurizer (paired bootstrap on shared queries)
    sig["our_svd_vs"] = {"within_creator": {}, "within_recipe": {}}
    if "our_svd" in cos_cache:
        for grp_name, grp in (("within_creator", groups), ("within_recipe", recipe_groups)):
            for name, cos in cos_cache.items():
                if name == "our_svd":
                    continue
                sig["our_svd_vs"][grp_name][name] = paired_bootstrap_gt(
                    cos_cache["our_svd"], cos, y, np.asarray(grp), n_boot=2000, seed=0)
    results["significance"] = sig
    cos_cache.clear()
    gc.collect()

    # ---- verdict logic (advisory; the human reads it) ----
    cc = results["cross_creator_cv"]
    wc = results["within_creator_mAP"]
    wr = results["within_recipe_mAP"]
    def num(x): return x if isinstance(x, float) else float("nan")
    verdict = []
    if "POC-1a" in mode:
        verdict.append("APPARATUS DEBUG ONLY — not a gate (need blind human labels + n>=300 for POC-1b).")
        our_p = (sig["within_creator"].get("our_svd") or {}).get("p_value")
        if our_p is not None and our_p <= 0.05:
            verdict.append(f"✓ pipeline runs; our_svd within-creator retrieval is above chance "
                           f"(perm p={our_p}) — apparatus sanity OK (NOT a gate result).")
        if results.get("cv_is_cross_creator") is False:
            verdict.append("⚠ too few creators for grouped CV → cross_creator_cv fell back to "
                           "NON-grouped CV (same creator in train+test); treat that number as leaky.")
        if results["errors"]:
            verdict.append(f"⚠ featurizer(s) crashed: {', '.join(results['errors'])}.")
    else:
        # SIGNIFICANCE-BASED GATE (no hard-coded margins). Using the permutation nulls + paired
        # bootstraps computed above, applied SYMMETRICALLY to BOTH the within-creator and within-recipe
        # groupings (the old code only refereed leakage within-creator — an asymmetry a reviewer flags).
        P_SIG = 0.01          # a featurizer is "above chance" iff permutation p ≤ this
        P_LEAK = 0.01         # a nuisance control "leaks" iff it is itself above chance at this p
        P_BEAT = 0.95         # our_svd "beats" a baseline iff paired-bootstrap P(Δ>0) ≥ this
        BASELINES = ("spectral_stat", "raw_ab", "w2t_svd", "product_sketch", "norm_only")
        # pure-nuisance leakage referee (NOT norm-inclusive: ΔW-norm is partly real signal — see A2).
        REFEREE = ("rank_leak_CONTROL", "recipe_fp_nonorm_CONTROL")

        if results["errors"]:
            verdict.append(f"⚠ {len(results['errors'])} featurizer(s) CRASHED ({', '.join(results['errors'])}) "
                           f"— NOT a clean gate; a crash is not 'didn't beat'. Fix before trusting the verdict.")

        def p_of(grp, name):
            return (sig[grp].get(name) or {}).get("p_value")
        def beat_p(grp, name):
            return (sig["our_svd_vs"][grp].get(name) or {}).get("p_a_gt_b")

        gate_ok = True
        for grp in ("within_creator", "within_recipe"):
            our_p = p_of(grp, "our_svd")
            our_sig = our_p is not None and our_p <= P_SIG
            leaks = [k for k in REFEREE if (p_of(grp, k) is not None and p_of(grp, k) <= P_LEAK)]
            beaten = {k: beat_p(grp, k) for k in BASELINES if beat_p(grp, k) is not None}
            not_beaten = [k for k, pv in beaten.items() if pv < P_BEAT]
            if leaks:
                verdict.append(f"⚠ [{grp}] LEAKAGE: nuisance control(s) {leaks} are themselves above chance "
                               f"(p≤{P_LEAK}) — the split leaks recipe/rank; fix before trusting.")
                gate_ok = False
            if not our_sig:
                verdict.append(f"✗ [{grp}] our_svd not above chance (perm p={our_p}).")
                gate_ok = False
            if not_beaten:
                verdict.append(f"✗ [{grp}] our_svd does not clearly beat {not_beaten} "
                               f"(paired-bootstrap P(Δ>0) < {P_BEAT}).")
                gate_ok = False
        # norm-inclusive recipe control reported as a DIAGNOSTIC only (A2)
        nd_p = p_of("within_creator", "recipe_fp_CONTROL")
        if nd_p is not None and nd_p <= P_LEAK:
            verdict.append("ℹ diagnostic: norm-inclusive recipe control is above chance while the norm-free "
                           "referee governs the gate — consistent with ΔW-norm carrying some real semantic "
                           "signal (not counted as leakage; see A2).")
        if gate_ok:
            verdict.append("✓ GATE PASSED: our_svd is above chance AND beats every baseline (paired bootstrap) "
                           "within BOTH creator and recipe groupings, and all pure-nuisance controls are at "
                           "chance. Significance-tested, not margin-thresholded.")
        else:
            verdict.append("✗ gate not passed (see specifics above). Significance-tested verdict.")
    results["verdict"] = verdict
    print("\nVerdict:")
    for v in verdict:
        print("  " + v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
