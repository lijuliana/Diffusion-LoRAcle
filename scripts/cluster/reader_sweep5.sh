#!/usr/bin/env bash
# Sweep #5: product_sketch reader, 8 arms, with both gates enforced in the launcher.
#
# Sweep #4 trained to epoch 2.9 of 3 and would have died at FINAL eval on
# `UnboundLocalError: local variable 'slot' referenced before assignment` in cross_lora_control,
# losing every arm's results AND its checkpoint, both of which are written after that call. The
# variable was added with the graded metrics and never initialised, so it could only fire after a
# full training run. A lint gate costs a second and makes that impossible to launch.
#
# Two gates, both fatal:
#   LINT      pyflakes over the training script; any undefined name aborts.
#   PREFLIGHT scripts/preflight.py over the token cache; anything but PASS aborts.
set -uo pipefail
MAN="${1:?manifest}"
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep5
WARM=ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120

echo "[gate 1/2] lint"
python -m pyflakes scripts/train_reader.py 2>&1 | grep -E "undefined name|referenced before" && {
  echo "LINT FAILED — not launching"; exit 1; }
echo "  no undefined names"

echo "[gate 2/2] preflight"
python -u scripts/preflight.py --token-cache data/tokens_psketch --manifest "$MAN" \
  > results/sweep5/preflight.log 2>&1
grep -q "PREFLIGHT PASSED" results/sweep5/preflight.log || {
  echo "PREFLIGHT FAILED — not launching"; tail -20 results/sweep5/preflight.log; exit 1; }
echo "  preflight passed on data/tokens_psketch"

( while true; do
    gcloud storage rsync -r results/sweep5/ gs://ditloracle-corpus/reader/sweep5/ >/dev/null 2>&1
    sleep 300
  done ) & SYNC=$!

run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_psketch --max-tokens 400 --lr 3e-5 \
    --out "results/sweep5/${name}.json" "$@" > "results/sweep5/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"; }

R16="--interpreter-rank 16 --lora-alpha 16"
# epoch ladder bracketing LoRAcle's ~237-step budget (ours is 186 steps/epoch)
run 0 ps_warm_e1  $R16 --epochs 1  --warm-start $WARM
run 1 ps_warm_e3  $R16 --epochs 3  --warm-start $WARM
run 2 ps_warm_e6  $R16 --epochs 6  --warm-start $WARM
# isolates warm start from optimisation budget
run 3 ps_cold_e3  $R16 --epochs 3
# capacity
run 4 ps_warm_r32_e3 --interpreter-rank 32 --lora-alpha 32 --epochs 3 --warm-start $WARM
# CONTROLS at matched epochs, lr and token budget. Both are needed: shuffled keeps token statistics
# and destroys the pairing; no-injection removes the tokens entirely.
run 5 ps_CONTROL_shuffled_e3 $R16 --epochs 3 --warm-start $WARM --shuffle-tokens
run 6 ps_CONTROL_noinject_e3 $R16 --epochs 3 --warm-start $WARM --no-injection
run 7 ps_CONTROL_shuffled_e6 $R16 --epochs 6 --warm-start $WARM --shuffle-tokens
wait
kill $SYNC 2>/dev/null
gcloud storage rsync -r results/sweep5/ gs://ditloracle-corpus/reader/sweep5/ 2>&1 | tail -1
echo "=== sweep5 complete ==="
