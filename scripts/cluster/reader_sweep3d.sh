#!/usr/bin/env bash
# Sweep #3d: run LoRAcle's actual regime instead of inventing one.
#
# Reading their shipped config rather than tuning against our own failures changes three things:
#
#   TOKENS. They inject n_direction_tokens=4480 per adapter (svd k16 x mag7 x 40 layers) at
#   max_length 5500. Our analogue is 16 directions x 25 klein blocks = 400. We were feeding 128,
#   i.e. 32% of each adapter, and before the round-robin fix always the same 16 modules of 50.
#   This sweep feeds all 400 and turns on gradient checkpointing to afford it.
#
#   EPOCHS. They train ONE epoch over ~1900 examples, about 237 optimizer steps. Sweep #3b/#3c ran
#   10-25 epochs, which at our 1490 examples is 1860-4650 steps, 8-20x their budget. Sweep #2 was
#   6x under; #3b/#3c were an order over. This sweep brackets THEIR step count.
#
#   CONTROLS DURING TRAINING. cross_lora_eval_every_epochs=0.1 in their config. Running the control
#   only at the end is how two sweeps burned complete runs before revealing they had fit nothing.
#   train_reader.py now prints train-accuracy and the shuffled-token control every 0.1 epoch, so a
#   setup that is not reading weights is visible within minutes.
#
# Calibration, also from their numbers: a fully-tuned LoRAcle at their scale, on their modality,
# with their warm start, scores ~30% mean across evals and ~12% rollout-mean. 30% is what working
# looks like. Nothing here should be judged against an intuition of 90%.
set -uo pipefail
MAN="${1:?manifest}"
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep3d
WARM=ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120

echo "[prep] token cache: $(ls data/tokens_projbank/*.pt 2>/dev/null | wc -l) adapters"

( while true; do
    gcloud storage rsync -r results/sweep3d/ gs://ditloracle-corpus/reader/sweep3d/ >/dev/null 2>&1
    sleep 300
  done ) & SYNC=$!

run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_projbank --max-tokens 400 --lr 3e-5 \
    --out "results/sweep3d/${name}.json" "$@" > "results/sweep3d/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"; }

R16="--interpreter-rank 16 --lora-alpha 16"
# epoch ladder BRACKETING LoRAcle's ~237 steps (ours: 186/epoch)
run 0 warm_e1   $R16 --epochs 1  --warm-start $WARM
run 1 warm_e3   $R16 --epochs 3  --warm-start $WARM
run 2 warm_e6   $R16 --epochs 6  --warm-start $WARM
run 3 warm_e12  $R16 --epochs 12 --warm-start $WARM
# isolates the warm start from the optimisation budget
run 4 cold_e3   $R16 --epochs 3
# capacity
run 5 warm_r32_e3 --interpreter-rank 32 --lora-alpha 32 --epochs 3 --warm-start $WARM
# CONTROLS at matched epochs, lr and token budget
run 6 CONTROL_shuffled_e3 $R16 --epochs 3 --warm-start $WARM --shuffle-tokens
run 7 CONTROL_noinject_e3 $R16 --epochs 3 --warm-start $WARM --no-injection
wait
kill $SYNC 2>/dev/null
gcloud storage rsync -r results/sweep3d/ gs://ditloracle-corpus/reader/sweep3d/ 2>&1 | tail -1
echo "=== sweep3d complete ==="
