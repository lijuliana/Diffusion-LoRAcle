#!/usr/bin/env bash
# Launch one mint shard as a detached, restart-on-preemption loop.
#
# LAUNCH THIS AS A SYSTEM-SCOPE SYSTEMD UNIT. nohup, setsid, and `systemd-run --user` all died when
# the ssh session closed (silently, mid-model-download, leaving no traceback and no manifest): with
# `Linger=no` the per-user systemd manager is torn down on logout and takes its children with it.
# What works:
#   sudo loginctl enable-linger $USER
#   sudo systemd-run --unit=mintshardN --collect --uid=$(id -u) --gid=$(id -g) \
#        --setenv=HOME=$HOME --working-directory=$HOME/mint/Diffusion-LoRAcle \
#        /bin/bash $HOME/mint/Diffusion-LoRAcle/scripts/cluster/run_shard.sh N 8 gate 12
# Check with `systemctl is-active mintshardN`, not by grepping ps.
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
