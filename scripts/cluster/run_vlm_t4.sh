#!/usr/bin/env bash
# Run the VLM stages (sensitive triage + benign drafting) on the Azure T4 box (WORKING_NORMS §1b)
# instead of a Forge p5 node. T4 = 16GB fp16-only → Qwen2.5-VL-7B must load in 4-bit (bitsandbytes).
#
# Differences from run_vlm.sh (Forge): no kubectl — plain ssh/rsync; images are rsynced UP from the
# laptop (the box has no CivitAI key by design — keep secrets off it); results rsynced BACK.
# Privacy (§2): repo dir on the box is a generic name, no project codename in paths.
#
# Prereqs (laptop): image download complete-ish (assets/corpus/images), VM started:
#   az vm start --resource-group rg-ditloracle-swedencentral --name ditloracle-t4-caption
# Usage:  bash scripts/cluster/run_vlm_t4.sh
set -euo pipefail

BOX="juliana@20.240.250.7"
RDIR="~/oss-caption"                     # generic name on the box (privacy §2)

echo "== 0. box reachable + GPU present =="
ssh -o ConnectTimeout=10 "${BOX}" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || {
  echo "box unreachable — start it: az vm start -g rg-ditloracle-swedencentral -n ditloracle-t4-caption"; exit 1; }

echo "== 1. sync code (repo files only; no secrets, no weights) =="
ssh "${BOX}" "mkdir -p ${RDIR}"
rsync -az --delete \
  ditloracle scripts pyproject.toml \
  "${BOX}:${RDIR}/"

echo "== 2. sync manifest + images up (~856MB first time; resumable) =="
ssh "${BOX}" "mkdir -p ${RDIR}/assets/corpus"
rsync -az assets/corpus/manifest_civitai_dl.json "${BOX}:${RDIR}/assets/corpus/"
rsync -az assets/corpus/images "${BOX}:${RDIR}/assets/corpus/"

echo "== 3. launch triage + drafting in the box venv (nohup — survives ssh drop) =="
ssh "${BOX}" "cd ${RDIR} && source ~/venv/bin/activate && pip -q install bitsandbytes qwen-vl-utils >/dev/null 2>&1;
  export PYTHONPATH=.
  nohup bash -c '
    python scripts/route_sensitive.py --backend qwen --quantization 4bit --out assets/corpus/sensitive_routing.json &&
    python scripts/vlm_draft_benign.py --backend qwen --quantization 4bit --out assets/corpus/vlm_drafts.json
  ' > vlm_run.log 2>&1 &
  echo started; sleep 3; tail -3 vlm_run.log || true"

echo "
Running on the T4 box. Monitor:   ssh ${BOX} 'tail -f ${RDIR#\~/}/vlm_run.log'
Pull results back when done:
  rsync -az ${BOX}:${RDIR}/assets/corpus/vlm_drafts.json        assets/corpus/
  rsync -az ${BOX}:${RDIR}/assets/corpus/sensitive_routing.json assets/corpus/
THEN deallocate (~\$1.20/hr while up):
  az vm deallocate -g rg-ditloracle-swedencentral -n ditloracle-t4-caption
"
