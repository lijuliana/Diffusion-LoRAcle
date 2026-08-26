#!/usr/bin/env bash
# Mint the NeurIPS-workshop corpus shard: 60 concepts x 6 replicates, recipe-varied.
#
# Why a recipe-VARIED corpus and not more of the gate set. The POC-M gate clamps the recipe by
# construction, which is what makes it causal — but it also makes the rank/recipe leakage control
# DEGENERATE: rank and module set are constant, so the control sits at chance no matter what, and
# carries no evidential weight (the gate harness says so in its own verdict text). The claim "the
# encoder reads semantics, not the recipe" therefore cannot be tested on the gate set at all. Here
# every concept is minted under all 6 entries of RECIPE_POOL (a complete block design, so measured
# concept<-recipe leakage is exactly 0.0), across ranks 8/16/32/64/128 — so the control becomes real
# and rank-invariance is measured over 5 ranks instead of 3.
#
# Seeds from the bucket first: the compute SA finally has storage.objectAdmin (2026-08-23), so a fresh
# box can skip every organism that already exists corpus-wide instead of re-minting it.
#
# Pair with autostop or the box bills until someone notices:
#   sudo systemd-run --unit=autostop --collect --setenv=HOME=$HOME /bin/bash $HOME/autostop.sh mintwork
#
# Usage (on the box):  bash run_workshop.sh <shard_index> <n_shards>
set -uo pipefail

I="${1:?shard index}"; N="${2:?n shards}"
REPO="$HOME/mint/Diffusion-LoRAcle"
BUCKET="${BUCKET:-gs://ditloracle-corpus}"

cd "$REPO"
source "$HOME/mint/venv/bin/activate"
export PYTHONPATH=.
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$REPO/assets/organisms/weights"
gcloud storage rsync -r "$BUCKET/organisms/weights/" "$REPO/assets/organisms/weights/" 2>&1 | tail -1
echo "seeded $(ls "$REPO"/assets/organisms/weights/*.safetensors 2>/dev/null | wc -l) existing adapters"

# 150 concepts x 6 replicates = 959 organisms, ~639 GPU-hours, ~40 h across 16 boxes.
# Sized against the 2-day budget, and enlarged from 60 concepts because LoRAcles reports performance
# scaling smoothly with corpus size up to 100K adapters; at 10^2 we are far below the regime where a
# weight-space reader is known to work, so corpus size is the most valuable thing GPU time can buy.
# Safe to enlarge mid-run: generate_concepts(n) is prefix-stable, so the 60-concept set is a strict
# subset of the 150-concept set and every adapter already minted still belongs to the plan.
# DO NOT discard this command's output. Suppressing it cost the first launch attempt: the box's repo
# predated the --n-concepts flag, argparse exited non-zero, no batch manifest was written, and
# mint_run then died on a missing file with the real cause already thrown away. Fail loudly instead.
if ! python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 6 --n-concepts 150 --n-images 12; then
  echo "FATAL: mint_corpus.py failed — is this box's code current? (needs the generative taxonomy)" >&2
  exit 1
fi
if [ ! -f assets/organisms/configs/batch_manifest.json ]; then
  echo "FATAL: no batch manifest after mint_corpus.py" >&2; exit 1
fi

# no --split: mint train AND test (test = held-out families, needed for the generalization split).
# Already-minted gate organisms are skipped because the seed above made the check corpus-wide.
python -u scripts/mint_run.py \
  --batch assets/organisms/configs/batch_manifest.json \
  --plan assets/organisms/mint_plan.json \
  --out "assets/organisms/minted_workshop_shard${I}.json" \
  --aitk-dir "$HOME/mint/ai-toolkit" \
  --shard "$I/$N" --n-images 12
echo "=== workshop shard $I finished rc=$? ==="
