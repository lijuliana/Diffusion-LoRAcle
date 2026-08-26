#!/usr/bin/env bash
# Fill in the gate organisms that never got minted, on a box that already ran a shard.
#
# Context (2026-08-23): the 47-organism gate mint finished on Aug 14 but 9 organisms were never
# attempted — the box running shard 0 died before writing its manifest. The other 38 adapters exist.
# `mint_run` skips an organism whose weights are already on disk, so seeding this box from the bucket
# first makes that skip GLOBAL: this run trains ONLY what is genuinely missing.
#
# These boxes are STANDARD (not spot), so local weights are durable and the run does not need the
# 5-minute sync loop; a flush at the end is enough. Launch as a SYSTEM-SCOPE systemd unit with linger
# on, or it dies at ssh close (PROGRESS 2026-08-13, failure mode #2).
#
# Usage (on the box):  bash fill_missing.sh <shard_index> <n_shards>
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
echo "seeded $(ls "$REPO"/assets/organisms/weights/*.safetensors 2>/dev/null | wc -l) adapters from the bucket"

python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 6 --n-images 12 >/dev/null 2>&1

python -u scripts/mint_run.py \
  --batch assets/organisms/configs/batch_manifest.json \
  --plan assets/organisms/mint_plan.json \
  --out "assets/organisms/minted_fill_shard${I}.json" \
  --aitk-dir "$HOME/mint/ai-toolkit" \
  --split gate --shard "$I/$N" --n-images 12
echo "=== fill shard $I finished rc=$? ==="
