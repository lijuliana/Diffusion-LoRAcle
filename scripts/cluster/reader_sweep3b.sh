#!/usr/bin/env bash
# Sweep #3b: correct the learning rate, and use TRAINING accuracy as the go/no-go.
#
# Sweep #2's real defect was not the metric and not the corpus. It could not fit its own TRAINING
# set (0.005-0.013 across all eight arms). A model that has not fit the data it saw says nothing
# about whether LoRA weights carry a readable signal, so the pivot criterion must not be applied to it.
#
# Cause: sweep #2 ran lr 5e-6 for 3 epochs. LoRAcle's base is lr 3e-5, and their "alpha = rank, halve
# lr" rule is prescribed for a BIGGER interpreter (their rank-512 collapse); sweep #2 applied that
# softening to rank 8-64, which are small, and overshot it 6x. With 395 examples at batch 1 x accum 8
# that is 147 optimizer steps at 5e-6 = roughly one TENTH the total optimization LoRAcle performed
# (~237 steps at 3e-5). The reader barely left initialization.
#
# So this sweep centres on lr 3e-5 and varies lr and epochs as a 2-factor grid, holding the encoder at
# projbank. Controls sit at the MATCHED lr and epoch count.
#
# Read TRAIN accuracy first. If it stays at floor here too, the setup is still broken and no held-out
# number from it is interpretable.
set -uo pipefail
MAN="${1:?manifest}"
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep3b
WARM=ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120

echo "[prep] reusing token cache: $(ls data/tokens_projbank/*.pt 2>/dev/null | wc -l) tokens"

( while true; do
    gcloud storage rsync -r results/sweep3b/ gs://ditloracle-corpus/reader/sweep3b/ >/dev/null 2>&1
    sleep 300
  done ) & SYNC=$!

run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_projbank \
    --out "results/sweep3b/${name}.json" "$@" > "results/sweep3b/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"; }

R16="--interpreter-rank 16 --lora-alpha 16"
# --- epoch ladder at LoRAcle's own lr
run 0 warm_lr3e5_e10  $R16 --lr 3e-5 --epochs 10 --warm-start $WARM
run 1 warm_lr3e5_e25  $R16 --lr 3e-5 --epochs 25 --warm-start $WARM
# --- lr ladder at fixed 25 epochs (5e-6 retained as the sweep-2 reference point)
run 2 warm_lr1e5_e25  $R16 --lr 1e-5 --epochs 25 --warm-start $WARM
run 3 warm_lr5e6_e25  $R16 --lr 5e-6 --epochs 25 --warm-start $WARM
# --- separates "warm-start helps" from "more optimization helps"
run 4 cold_lr3e5_e25  $R16 --lr 3e-5 --epochs 25
# --- capacity at the corrected lr
run 5 warm_r32_lr3e5_e25 --interpreter-rank 32 --lora-alpha 32 --lr 3e-5 --epochs 25 --warm-start $WARM
# --- CONTROLS at matched lr AND matched epochs
run 6 CONTROL_shuffled_lr3e5_e25 $R16 --lr 3e-5 --epochs 25 --warm-start $WARM --shuffle-tokens
run 7 CONTROL_noinject_lr3e5_e25 $R16 --lr 3e-5 --epochs 25 --warm-start $WARM --no-injection
wait
kill $SYNC 2>/dev/null
gcloud storage rsync -r results/sweep3b/ gs://ditloracle-corpus/reader/sweep3b/ 2>&1 | tail -1
echo "=== sweep3b complete ==="
