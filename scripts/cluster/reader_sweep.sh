#!/usr/bin/env bash
# Eight reader configs, one per H100, run concurrently.
#
# This is an ABLATION, not a hyperparameter hunt. Peer review's sharpest structural point was that the
# cross-architecture / cross-modality bridge is called the project's biggest unknown and has no
# ablation attached to it. These eight arms are that ablation. Arms 7 and 8 are controls that MUST
# fail; a result that does not beat both of them is measuring the concept prior, not the weights.
#
# Usage (on the 8-GPU box):  bash reader_sweep.sh <manifest>
set -uo pipefail
MAN="${1:?manifest}"
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep data

# ---- Phase A: build the token caches ONCE, on GPU. This is the step whose absence cost the first
# attempt: eight arms each SVD-ing 20k modules on CPU, load average 815, every H100 idle.
if [ ! -f data/tokens_randorth/_vocab.json ]; then
  echo "[extract] random_orth ..."
  python -u scripts/extract_tokens.py --manifest "$MAN" --bridge random_orth \
    --out data/tokens_randorth --n-directions 16 --d-token 5120 --device cuda
fi
if [ ! -f data/tokens_projbank/_vocab.json ]; then
  echo "[extract] projbank ..."
  python -u scripts/extract_tokens.py --manifest "$MAN" --bridge projbank \
    --out data/tokens_projbank --n-directions 16 --d-token 5120 --device cuda
fi
echo "[extract] done: randorth=$(ls data/tokens_randorth/*.pt 2>/dev/null | wc -l) projbank=$(ls data/tokens_projbank/*.pt 2>/dev/null | wc -l)"


run () {   # gpu name extra_args...
  local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN" --model Qwen/Qwen3-14B --device cuda \
    --out "results/sweep/${name}.json" "$@" \
    > "results/sweep/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"
}

run 0 arm1_randbridge_noWS      --token-cache data/tokens_randorth --n-directions 1
run 1 arm2_projbank_noWS        --token-cache data/tokens_projbank --n-directions 1
run 2 arm3_projbank_warmstart   --token-cache data/tokens_projbank --n-directions 1 --warm-start ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120
run 3 arm4_learnedbridge        --token-cache data/tokens_randorth --n-directions 1 --learned-bridge
run 4 arm5_k1                   --token-cache data/tokens_projbank --n-directions 1
run 5 arm6_k16                  --token-cache data/tokens_projbank --n-directions 16
run 6 arm7_CONTROL_shuffled     --token-cache data/tokens_projbank --n-directions 1 --shuffle-tokens
run 7 arm8_CONTROL_noinject     --token-cache data/tokens_projbank --n-directions 1 --no-injection
wait
echo "=== SWEEP COMPLETE $(date) ==="
