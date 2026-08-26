#!/usr/bin/env bash
# Mint ONLY the gate organisms that were never attempted, from a pre-filtered batch manifest.
#
# Context (2026-08-23): the 47-organism gate mint finished Aug 14 but the box running shard 0 died
# before writing its manifest, so 9 organisms were never attempted. Rather than seed every box from
# the bucket to make `mint_run`'s skip-if-exists global, the caller builds a batch manifest holding
# EXACTLY the missing 9 (`assets/organisms/configs/batch_fill.json`) and ships it here. The box then
# needs no bucket credentials at all — which matters, because the default compute SA has no
# storage.objects.* on gs://ditloracle-corpus (that is why the original sync silently 403'd).
#
# These boxes are STANDARD, not spot, so local weights are durable and no 5-minute sync loop is
# needed; the caller flushes to the bucket at the end with its own token.
#
# Launch as a SYSTEM-SCOPE unit with linger on, or it dies at ssh close (PROGRESS 2026-08-13 #2):
#   sudo loginctl enable-linger $USER
#   sudo systemd-run --unit=mintfill --collect --uid=$(id -u) --gid=$(id -g) \
#        --setenv=HOME=$HOME --working-directory=$HOME /bin/bash $HOME/run_fill.sh 0 4
# Check with `systemctl is-active mintfill`, never by grepping ps.
#
# ALWAYS pair it with the autostop watcher, or the box bills until a human notices (that cost ~$760
# across five boxes after the Aug-14 mint finished):
#   sudo systemd-run --unit=autostop --collect --setenv=HOME=$HOME /bin/bash $HOME/autostop.sh mintfill
#
# Usage (on the box):  bash run_fill.sh <shard_index> <n_shards>
set -uo pipefail

I="${1:?shard index}"; N="${2:?n shards}"
REPO="$HOME/mint/Diffusion-LoRAcle"

cd "$REPO"
source "$HOME/mint/venv/bin/activate"
export PYTHONPATH=.
export HF_HUB_DISABLE_TELEMETRY=1

cp "$HOME/batch_fill.json" "$REPO/assets/organisms/configs/batch_fill.json"
# regenerate per-organism configs + plan (deterministic; same seeds as the original run)
python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 6 --n-images 12 >/dev/null 2>&1

python -u scripts/mint_run.py \
  --batch assets/organisms/configs/batch_fill.json \
  --plan assets/organisms/mint_plan.json \
  --out "assets/organisms/minted_fill_shard${I}.json" \
  --aitk-dir "$HOME/mint/ai-toolkit" \
  --split gate --shard "$I/$N" --n-images 12
echo "=== fill shard $I finished rc=$? ==="
