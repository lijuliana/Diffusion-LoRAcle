#!/usr/bin/env bash
# Sweep #2: interpreter CAPACITY, after sweep #1 collapsed to a degenerate fixed point.
#
# Sweep #1 ran LoRAcle's shipped config (rank 256 / alpha 32 / lr 3e-5) and every arm — including both
# controls — emitted the SAME sentence for every adapter. That is the failure their CLAUDE.md
# describes: "rank-512 with the same lr/alpha collapses into a degenerate fixed-point for every
# organism within ~1 epoch. The fix is alpha=rank and lr=5e-5." Their config is tuned for ~1900
# examples with a warm-start; we ran 1.03e9 trainable parameters against 395. Collapse is the expected
# outcome, and a collapsed interpreter tests nothing about whether weights carry signal.
#
# So this sweep varies CAPACITY, holding the encoder fixed at the best-grounded option (projbank).
# Controls are retained at the most promising capacity so "beats the control" stays answerable.
set -uo pipefail
MAN="${1:?manifest}"
REPO="$HOME/mint/Diffusion-LoRAcle"
cd "$REPO"; source "$HOME/mint/venv/bin/activate"; export PYTHONPATH=.
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"
mkdir -p results/sweep2

# ---- Phase A: refresh corpus + tokens INSIDE the unit.
# Previously the launcher did this inside an `ssh --command`, so the rsync and extraction were held by
# the ssh connection and would die with it — the same "orchestration living in a terminal" failure that
# nearly dropped a mint shard earlier today. Anything that must outlive the connection belongs in the
# systemd unit, so it lives here now and the launcher only has to start the unit and return.
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
echo "[prep] extending projbank token cache ..."
python -u scripts/extract_tokens.py --manifest "$MAN" --bridge projbank \
  --out data/tokens_projbank --n-directions 16 --d-token 5120 --device cuda 2>&1 | tail -2
echo "[prep] tokens: $(ls data/tokens_projbank/*.pt 2>/dev/null | wc -l)"


run () { local gpu=$1; local name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u scripts/train_reader.py \
    --manifest "$MAN" --model Qwen/Qwen3-14B --device cuda \
    --token-cache data/tokens_projbank \
    --out "results/sweep2/${name}.json" "$@" > "results/sweep2/${name}.log" 2>&1 &
  echo "  gpu$gpu -> $name"; }

# alpha = rank throughout (scaling 1.0), lr halved, per their stated fix.
run 0 r8_lr5e6      --interpreter-rank 8   --lora-alpha 8   --lr 5e-6  --epochs 3
run 1 r16_lr5e6     --interpreter-rank 16  --lora-alpha 16  --lr 5e-6  --epochs 3
run 2 r32_lr5e6     --interpreter-rank 32  --lora-alpha 32  --lr 5e-6  --epochs 3
run 3 r64_lr5e6     --interpreter-rank 64  --lora-alpha 64  --lr 5e-6  --epochs 3
run 4 r16_lr1e5     --interpreter-rank 16  --lora-alpha 16  --lr 1e-5  --epochs 3
run 5 r16_warmstart --interpreter-rank 16  --lora-alpha 16  --lr 5e-6  --epochs 3 \
                    --warm-start ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120
run 6 r16_CONTROL_shuffled  --interpreter-rank 16 --lora-alpha 16 --lr 5e-6 --epochs 3 --shuffle-tokens
run 7 r16_CONTROL_noinject  --interpreter-rank 16 --lora-alpha 16 --lr 5e-6 --epochs 3 --no-injection
wait
echo "=== SWEEP2 COMPLETE $(date) ==="
