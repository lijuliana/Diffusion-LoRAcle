#!/usr/bin/env bash
# Sweep #6: extend the epoch ladder, with a control paired to every real arm.
#
# Sweep #5 found the first movement off the floor. At 6 epochs on product_sketch tokens:
#   train 0.042 vs control 0.013 | held-out 3/84 vs control 0/84 | rank 0.484 vs control 0.510
# Three metrics agree in direction, and Fisher exact on 3/84 vs 0/84 gives p=0.123, so none of it is
# significant at this n. Two things raise power and nothing else needs changing:
#
#   MORE EPOCHS. 1 and 3 epochs sat at the floor (train 0.010-0.012); 6 moved (0.042). If the effect
#   is real it should grow at 12 and 25. Earlier 25-epoch sweeps saw nothing because they ran on
#   projbank tokens, which measure 0/98 concept.
#   MORE HELD-OUT. Held-out n is the count of concepts with >=3 adapters. The corpus has grown from
#   625 to ~764, so the manifest and token cache are rebuilt here before training.
#
# Every real arm has a shuffled control at MATCHED epochs, so the comparison never comes from an
# arm trained longer than the thing it is compared against.
set -uo pipefail
MAN_OUT=assets/organisms/provisional_workshop.json
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep6
WARM=ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120

echo "[gate 1/3] lint"
python -m pyflakes scripts/train_reader.py 2>&1 | grep -E "undefined name|referenced before" && {
  echo "LINT FAILED"; exit 1; }
echo "  clean"

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

echo "[prep] extracting product_sketch tokens for the enlarged corpus"
python -u scripts/extract_tokens.py --manifest "$MAN_OUT" --bridge product_sketch \
  --out data/tokens_psketch_v2 --d-token 5120 --device cpu 2>&1 | tail -2
echo "  tokens: $(ls data/tokens_psketch_v2/*.pt 2>/dev/null | wc -l)"

echo "[gate 2/3] preflight"
python -u scripts/preflight.py --token-cache data/tokens_psketch_v2 --manifest "$MAN_OUT" \
  > results/sweep6/preflight.log 2>&1
grep -q "PREFLIGHT PASSED" results/sweep6/preflight.log || {
  echo "PREFLIGHT FAILED"; tail -20 results/sweep6/preflight.log; exit 1; }
grep -E "carries concept|held-out large enough" results/sweep6/preflight.log

echo "[gate 3/3] disk"
avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
[ "$avail" -lt 60 ] && { echo "ONLY ${avail}G FREE — not launching"; exit 1; }
echo "  ${avail}G free"

( while true; do
    gcloud storage rsync -r results/sweep6/ gs://ditloracle-corpus/reader/sweep6/ >/dev/null 2>&1
    sleep 300
  done ) & SYNC=$!

run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN_OUT" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_psketch_v2 --max-tokens 400 --lr 3e-5 \
    --out "results/sweep6/${name}.json" "$@" > "results/sweep6/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"; }

R16="--interpreter-rank 16 --lora-alpha 16"
run 0 e6_real   $R16 --epochs 6  --warm-start $WARM
run 1 e6_CTRL   $R16 --epochs 6  --warm-start $WARM --shuffle-tokens
run 2 e12_real  $R16 --epochs 12 --warm-start $WARM
run 3 e12_CTRL  $R16 --epochs 12 --warm-start $WARM --shuffle-tokens
run 4 e25_real  $R16 --epochs 25 --warm-start $WARM
run 5 e25_CTRL  $R16 --epochs 25 --warm-start $WARM --shuffle-tokens
run 6 e12_r32_real --interpreter-rank 32 --lora-alpha 32 --epochs 12 --warm-start $WARM
run 7 e12_noinject_CTRL $R16 --epochs 12 --warm-start $WARM --no-injection
wait
kill $SYNC 2>/dev/null
gcloud storage rsync -r results/sweep6/ gs://ditloracle-corpus/reader/sweep6/ 2>&1 | tail -1
echo "=== sweep6 complete ==="
