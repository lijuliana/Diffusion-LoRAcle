#!/usr/bin/env bash
# Sweep #7: the capacity test we believed we had already run, and had not.
#
# PeftModel.from_pretrained replaces the LoRA configured before it, so --interpreter-rank has been
# inert on every warm-started arm since sweep #2 and all of them trained at the warm start's rank 256
# (1,028,526,080 trainable parameters; rank 16 would be 64.2M). The capacity ladder in sweeps #2 and
# #3 compared ranks 8/16/32/64 across arms that were all the same rank-256 model, and the remedy for
# sweep #1's collapse, which was to reduce capacity, never reached the arms it targeted.
#
# A warm start cannot be combined with a chosen rank, because it dictates the rank. So the only way
# to obtain rank 16 is to start cold. That introduces one confound and it must be stated: a cold
# interpreter has to learn output FORMAT as well as content, where a warm start supplies the format.
# If this arm fails, capacity and missing format skill are not separable from it alone.
#
# Matched to the rank-256 e12_real already running: same 12 epochs, same lr, same token budget, same
# corpus. The comparison is capacity at fixed optimisation.
#
# The one existing genuine rank-16 arm, sweep #5's ps_cold_e3, sat at the floor at THREE epochs
# (train 0.012, held-out 1/84). The warm arms only moved at six, so three was too few to conclude.
set -uo pipefail
MAN=assets/organisms/provisional_workshop.json
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep7

echo "[gate 1/3] lint"
python -m pyflakes scripts/train_reader.py 2>&1 | grep -E "undefined name|referenced before" && {
  echo "LINT FAILED"; exit 1; }
echo "  clean"

echo "[gate 2/3] preflight"
python -u scripts/preflight.py --token-cache data/tokens_psketch_v2 --manifest "$MAN" \
  > results/sweep7/preflight.log 2>&1
grep -q "PREFLIGHT PASSED" results/sweep7/preflight.log || {
  echo "PREFLIGHT FAILED"; tail -20 results/sweep7/preflight.log; exit 1; }
grep -E "carries concept|held-out large enough" results/sweep7/preflight.log

echo "[gate 3/3] disk"
avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
[ "$avail" -lt 60 ] && { echo "ONLY ${avail}G FREE — not launching"; exit 1; }
echo "  ${avail}G free"

( while true; do
    gcloud storage rsync -r results/sweep7/ gs://ditloracle-corpus/reader/sweep7/ >/dev/null 2>&1
    sleep 300
  done ) & SYNC=$!

run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_psketch_v2 --max-tokens 400 --lr 3e-5 \
    --interpreter-rank 16 --lora-alpha 16 --epochs 12 \
    --out "results/sweep7/${name}.json" "$@" > "results/sweep7/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"; }

# No --warm-start on either arm, so rank 16 is actually applied. Verify in the log that trainable
# parameters read ~64.2M and not 1,028M before believing any number from these arms.
run 0 cold_r16_e12_real
run 1 cold_r16_e12_CTRL --shuffle-tokens
wait
kill $SYNC 2>/dev/null
gcloud storage rsync -r results/sweep7/ gs://ditloracle-corpus/reader/sweep7/ 2>&1 | tail -1
echo "=== sweep7 complete ==="
