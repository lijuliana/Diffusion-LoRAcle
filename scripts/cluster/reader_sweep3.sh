#!/usr/bin/env bash
# Sweep #3: does the reader carry PARTIAL signal, and is it simply undertrained?
#
# Sweep #2 fixed the mode collapse (varied generations) but every arm sat at 0.000-0.029 exact-match
# against a 0.029 nearest-neighbour baseline, i.e. 1-2 of 70. Two things were wrong with that readout:
#
#   1. THE METRIC. concept_hit() demands verbatim substring match of a median-8-word compositional
#      name from 150 candidates, so a reader getting three slots of four scored exactly the same as
#      one emitting nonsense. train_reader.py now also reports slot_credit (fraction of discriminative
#      slots hit) and a normalised retrieval rank (0.5 = chance). Slot values with >25% document
#      frequency are excluded, because the family slot takes 3 values covering 41% of the corpus and
#      was handing free credit to wrong answers.
#   2. NO CHECKPOINTS. train_reader.py never persisted the reader, so all 8 sweep-2 arms had to be
#      retrained to be re-scored. It now writes <out>.reader.pt.
#
# Sweep #2 also ran 3 epochs over ~395 examples. LoRAcle trained on ~1900. Undertraining is a distinct
# hypothesis from "weights carry no signal", so epochs are varied here with controls at the MATCHED
# epoch count — otherwise a longer-trained arm beating a shorter-trained control proves nothing.
set -uo pipefail
MAN="${1:?manifest}"
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep3
WARM=ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120

# ---- Phase A: prep INSIDE the unit (never in `ssh --command`; it dies with the connection).
echo "[prep] syncing corpus ..."
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
# The token cache did not survive the box's 08:20/09:10 reboots, so it is rebuilt here.
echo "[prep] rebuilding projbank token cache ..."
python -u scripts/extract_tokens.py --manifest "$MAN" --bridge projbank \
  --out data/tokens_projbank --n-directions 16 --d-token 5120 --device cuda 2>&1 | tail -2
echo "[prep] tokens: $(ls data/tokens_projbank/*.pt 2>/dev/null | wc -l)"

# ---- Phase B: push results + checkpoints to the bucket as they land, so a reboot cannot erase
# another full sweep the way it erased the token cache.
( while true; do
    gcloud storage rsync -r results/sweep3/ gs://ditloracle-corpus/reader/sweep3/ >/dev/null 2>&1
    sleep 300
  done ) & SYNC=$!

run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_projbank \
    --out "results/sweep3/${name}.json" "$@" > "results/sweep3/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"; }

B="--interpreter-rank 16 --lora-alpha 16 --lr 5e-6"
# epoch ladder on the best sweep-2 arm (warm-start, 0.029)
run 0 warm_e3    $B --epochs 3  --warm-start $WARM
run 1 warm_e10   $B --epochs 10 --warm-start $WARM
run 2 warm_e25   $B --epochs 25 --warm-start $WARM
# cold-start at the long setting, to separate "warm-start helps" from "more epochs help"
run 3 cold_e25   $B --epochs 25
# capacity at the long setting
run 4 warm_r32_e25 --interpreter-rank 32 --lora-alpha 32 --lr 5e-6 --epochs 25 --warm-start $WARM
run 5 warm_e25_lr1e5 --interpreter-rank 16 --lora-alpha 16 --lr 1e-5 --epochs 25 --warm-start $WARM
# CONTROLS at the MATCHED epoch count (25), or the comparison is meaningless
run 6 CONTROL_shuffled_e25 $B --epochs 25 --warm-start $WARM --shuffle-tokens
run 7 CONTROL_noinject_e25 $B --epochs 25 --warm-start $WARM --no-injection
wait
kill $SYNC 2>/dev/null
gcloud storage rsync -r results/sweep3/ gs://ditloracle-corpus/reader/sweep3/ 2>&1 | tail -1
echo "=== sweep3 complete ==="
