#!/usr/bin/env bash
# Sweep #8: rerun the full table with the LEAK-FIXED adapter-level split.
#
# Sweeps 6 and 7 split at the (adapter, question) EXAMPLE level: each "held-out" adapter's other
# three question-examples, with identical weight tokens, were in training, so their held-out numbers
# measure held-out-QUESTION accuracy on seen adapters. train_reader.py now splits at the ADAPTER
# level (all four examples of the held-out adapter are held out together), and the cross-LoRA
# intervention evaluates every pair instead of a 24-pair subsample. Same corpus, tokens, recipes,
# and hyperparameters as sweep #6; the only changes are the split and the intervention n.
#
# 10 arms on 8 GPUs; shorter arms chain: worst path is e25 (~32.5 h at ~1.3 h/epoch).
set -uo pipefail
MAN_OUT=assets/organisms/provisional_workshop.json
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep8
WARM=ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120

echo "[gate 1/4] lint"
python -m pyflakes scripts/train_reader.py 2>&1 | grep -E "undefined name|referenced before" && {
  echo "LINT FAILED"; exit 1; }
echo "  clean"

echo "[gate 2/4] split is adapter-level (the point of this sweep)"
python - <<'PY'
import inspect, re
src = open('scripts/train_reader.py').read()
assert 'by_org' in src and 'Split at the ADAPTER level' in src, 'train_reader.py lacks the fixed split'
print('  fixed split present')
PY

echo "[prep] refreshing corpus + manifest"
gcloud storage rsync -r gs://ditloracle-corpus/organisms/weights/ assets/organisms/weights/ 2>&1 | tail -1
python - <<'PY'
import json, pathlib
plan = json.loads(pathlib.Path('assets/organisms/mint_plan_workshop.json').read_text())
w = pathlib.Path('assets/organisms/weights'); have = {p.stem for p in w.glob('*.safetensors')}
orgs = [dict(o, weights_path=str(w / f"{o['organism_id']}.safetensors"))
        for o in plan['organisms'] if o['organism_id'] in have]
pathlib.Path('assets/organisms/provisional_workshop.json').write_text(
    json.dumps({'n_organisms': len(orgs), 'organisms': orgs}, indent=2))
print(f"[prep] manifest -> {len(orgs)} adapters")
PY

echo "[prep] extending product_sketch token cache"
python -u scripts/extract_tokens.py --manifest "$MAN_OUT" --bridge product_sketch \
  --out data/tokens_psketch_v2 --d-token 5120 --device cpu 2>&1 | tail -2
echo "  tokens: $(ls data/tokens_psketch_v2/*.pt 2>/dev/null | wc -l)"

echo "[gate 3/4] preflight"
python -u scripts/preflight.py --token-cache data/tokens_psketch_v2 --manifest "$MAN_OUT" \
  > results/sweep8/preflight.log 2>&1
grep -q "PREFLIGHT PASSED" results/sweep8/preflight.log || {
  echo "PREFLIGHT FAILED"; tail -20 results/sweep8/preflight.log; exit 1; }
grep -E "carries concept|held-out large enough" results/sweep8/preflight.log

echo "[gate 4/4] disk"
avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
[ "$avail" -lt 60 ] && { echo "ONLY ${avail}G FREE — not launching"; exit 1; }
echo "  ${avail}G free"

( while true; do
    gcloud storage rsync -r results/sweep8/ gs://ditloracle-corpus/reader/sweep8/ >/dev/null 2>&1
    sleep 300
  done ) & SYNC=$!

# run <gpu> <name> [extra flags...] — foreground inside a per-GPU chain
run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/train_reader.py \
    --manifest "$MAN_OUT" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_psketch_v2 --max-tokens 400 --lr 3e-5 \
    --out "results/sweep8/${name}.json" "$@" > "results/sweep8/${name}.log" 2>&1
  echo "  gpu$gpu done: $name ($(date -u +%H:%M))"; }

# Warm arms take the checkpoint's rank (256); passing --interpreter-rank 16 with a warm start
# now ABORTS (the guard added after sweep 6, which these flags were copied from). The r32 "repeat"
# arm keeps the override flag so it reproduces sweep 6's silently-256 twin explicitly.
COLD="--interpreter-rank 16 --lora-alpha 16 --epochs 12"

( run 0 e25_real  --epochs 25 --warm-start $WARM ) &
( run 1 e25_CTRL  --epochs 25 --warm-start $WARM --shuffle-tokens ) &
( run 2 e12_real  --epochs 12 --warm-start $WARM
  run 2 e6_real   --epochs 6  --warm-start $WARM ) &
( run 3 e12_CTRL  --epochs 12 --warm-start $WARM --shuffle-tokens
  run 3 e6_CTRL   --epochs 6  --warm-start $WARM --shuffle-tokens ) &
( run 4 e12_r32_real --interpreter-rank 32 --lora-alpha 32 --allow-rank-override --epochs 12 --warm-start $WARM ) &
( run 5 e12_noinject_CTRL --epochs 12 --warm-start $WARM --no-injection ) &
( run 6 cold_r16_e12_real $COLD ) &
( run 7 cold_r16_e12_CTRL $COLD --shuffle-tokens ) &
wait
kill $SYNC 2>/dev/null
gcloud storage rsync -r results/sweep8/ gs://ditloracle-corpus/reader/sweep8/ 2>&1 | tail -1
echo "=== sweep8 complete ==="
