"""Emit the mint-first corpus plan + ai-toolkit training configs (local; no GPU).

This is the entry point for the pivot (PLAN.md §6). It builds the full plan (capability taxonomy +
safety families + causal-gate matched sets), validates every counterfactual, and writes one trainer
config per organism plus a batch manifest the cluster launcher consumes. Nothing here trains; it is
the deterministic hand-off from local design to cluster minting.

Usage:
  python scripts/mint_corpus.py --base FLUX.1-dev --replicates 2
  python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 4 --plan-only
  python scripts/mint_corpus.py --base FLUX.2-klein-4B --n-concepts 600 --replicates 2

`--replicates` is DEPTH (organisms per concept) and `--n-concepts` is BREADTH (how many concepts
exist at all). POC-M wants depth on the curated 22; POC-C wants breadth, because replicates of a
fixed 22-concept set are near-duplicates and an open-language reader needs hundreds of concepts
(PLAN.md §6). Leaving `--n-concepts` off keeps the curated 22, which is the POC-M default.
"""

from __future__ import annotations

import argparse

from ditloracle.mint import corpus_plan, trainer_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the mint-first corpus plan + trainer configs.")
    ap.add_argument("--base", default="FLUX.1-dev",
                    help="base model label (FLUX.1-dev / FLUX.2-klein-4B)")
    ap.add_argument("--replicates", type=int, default=2,
                    help="capability organisms per concept (recipe-decorrelated)")
    ap.add_argument("--n-concepts", type=int, default=None,
                    help="total concepts: omit for the curated 22, or pass hundreds to expand the "
                         "taxonomy compositionally (the curated 22 stay the first 22)")
    ap.add_argument("--plan-out", default="assets/organisms/mint_plan.json")
    ap.add_argument("--config-dir", default="assets/organisms/configs")
    ap.add_argument("--n-images", type=int, default=12,
                    help="images per organism; MUST match mint_run --n-images (sets training steps)")
    ap.add_argument("--plan-only", action="store_true", help="skip writing trainer configs")
    a = ap.parse_args()

    plan = corpus_plan.write_plan(a.plan_out, base_model=a.base, replicates=a.replicates,
                                  n_concepts=a.n_concepts)
    print(f"mint plan -> {a.plan_out}")
    print(f"  base model    : {plan['base_model']}")
    print(f"  concepts      : {plan['n_concepts']}  x {plan['replicates']} replicates")
    print(f"  organisms     : {plan['n_organisms']}  "
          f"({plan['n_capability']} capability + {plan['n_safety']} safety + "
          f"{plan['split_tally']['gate']} in {plan['n_matched_sets']} gate matched-sets)")
    print(f"  split tally   : train={plan['split_tally']['train']}  "
          f"test={plan['split_tally']['test']}  gate={plan['split_tally']['gate']}")
    print(f"  held-out fams : {', '.join(plan['held_out_families'])}")
    if plan["errors"]:
        print(f"  ⚠ {len(plan['errors'])} VALIDATION ERRORS (fix before minting):")
        for e in plan["errors"][:20]:
            print(f"    ✗ {e}")
        return
    print("  ✓ all counterfactual matched sets validate")

    if a.plan_only:
        return
    summary = trainer_config.write_configs(plan, a.config_dir, n_images=a.n_images)
    print(f"trainer configs -> {summary['out_dir']}  "
          f"({summary['n_configs']} configs; {summary['n_needs_exact']} need exact-module diffusers path)")
    print(f"  batch manifest -> {summary['batch_manifest']}")


if __name__ == "__main__":
    main()
