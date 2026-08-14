#!/usr/bin/env bash
# Launch one mint shard as a detached, restart-on-preemption loop.
#
# Backgrounding straight from `gcloud compute ssh --command` did not survive the session closing
# (processes died mid-download with no traceback), so the miner is started from a script the box owns
# and re-execs itself if it dies — which also covers spot preemption of the training subprocess.
#
# Usage (on the box):  bash run_shard.sh <shard_index> <n_shards> <split> [n_images]
set -uo pipefail

I="${1:?shard index}"; N="${2:?n shards}"; SPLIT="${3:-gate}"; NIMG="${4:-12}"
REPO="$HOME/mint/Diffusion-LoRAcle"

cd "$REPO"
# shellcheck disable=SC1091
source "$HOME/mint/venv/bin/activate"
export PYTHONPATH=.
export HF_HUB_DISABLE_TELEMETRY=1

python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 6 --n-images "$NIMG" >/dev/null 2>&1

# mint_run is resumable (finished organisms are skipped), so re-running after a crash is safe and cheap
for attempt in 1 2 3 4 5; do
  echo "=== mint attempt $attempt (shard $I/$N) ===" >> "$HOME/mint/mint_shard.log"
  python -u scripts/mint_run.py \
    --batch assets/organisms/configs/batch_manifest.json \
    --plan assets/organisms/mint_plan.json \
    --out "assets/organisms/minted_${SPLIT}_shard${I}.json" \
    --aitk-dir "$HOME/mint/ai-toolkit" \
    --split "$SPLIT" --shard "$I/$N" --n-images "$NIMG" >> "$HOME/mint/mint_shard.log" 2>&1
  rc=$?
  echo "=== attempt $attempt exited rc=$rc ===" >> "$HOME/mint/mint_shard.log"
  [ $rc -eq 0 ] && break
  sleep 20
done
echo "=== shard $I finished ===" >> "$HOME/mint/mint_shard.log"
