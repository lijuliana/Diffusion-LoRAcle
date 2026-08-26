#!/usr/bin/env bash
# Sweep #4: the first reader run on an input that provably contains the answer.
#
# Every previous sweep trained on projbank tokens. scripts/preflight.py on that cache FAILS two
# checks: concept accuracy 1/98 (p=0.56), and a label-shuffle control scoring 0.031 against the
# true labels' 0.010, i.e. fitting real labels is no better than fitting nonsense. No learning
# rate, epoch count or corpus size could have rescued that.
#
# product_sketch tokens pass all 14 preflight checks:
#   concept 6/98 (p=1.83e-04) | label-shuffle collapses 0.061 -> 0.000 | concept 7.3x vs rank 3.8x
#
# Four arms only. The other four GPUs finish the projbank negative (warm_e1, warm_e3 and both
# controls at matched settings), which is a publishable result in its own right and should not be
# thrown away to buy parallelism here.
#
# Calibration: LoRAcle scores ~30% on their own task. A linear classifier gets 6/98 top-1 and 17/98
# top-5 from these tokens. Expect single digits, and judge against the controls, not against 30%.
set -uo pipefail
MAN="${1:?manifest}"
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep4
WARM=ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120

# Refuse to run if the tokens do not pass. The point of preflight is that a bad run cannot start.
python -u scripts/preflight.py --token-cache data/tokens_psketch --manifest "$MAN" \
  > results/sweep4/preflight.log 2>&1
if ! grep -q "PREFLIGHT PASSED" results/sweep4/preflight.log; then
  echo "PREFLIGHT FAILED — not launching"; tail -20 results/sweep4/preflight.log; exit 1
fi
echo "[prep] preflight passed on data/tokens_psketch"

( while true; do
    gcloud storage rsync -r results/sweep4/ gs://ditloracle-corpus/reader/sweep4/ >/dev/null 2>&1
    sleep 300
  done ) & SYNC=$!

run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_psketch --max-tokens 400 --lr 3e-5 \
    --out "results/sweep4/${name}.json" "$@" > "results/sweep4/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"; }

R16="--interpreter-rank 16 --lora-alpha 16"
run 2 ps_warm_e3   $R16 --epochs 3 --warm-start $WARM
run 3 ps_warm_e1   $R16 --epochs 1 --warm-start $WARM
run 4 ps_CONTROL_shuffled_e3 $R16 --epochs 3 --warm-start $WARM --shuffle-tokens
run 5 ps_CONTROL_noinject_e3 $R16 --epochs 3 --warm-start $WARM --no-injection
wait
kill $SYNC 2>/dev/null
gcloud storage rsync -r results/sweep4/ gs://ditloracle-corpus/reader/sweep4/ 2>&1 | tail -1
echo "=== sweep4 complete ==="
