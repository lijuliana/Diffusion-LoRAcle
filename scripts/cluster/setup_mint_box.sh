#!/usr/bin/env bash
# Provision a fresh GPU box for minting: ai-toolkit + our package + the base model.
# Idempotent — safe to re-run after a preemption or a failed step.
#
# Usage (on the box):  bash setup_mint_box.sh
set -euo pipefail

REPO_URL="https://github.com/lijuliana/Diffusion-LoRAcle.git"
BRANCH="recovery/mint-first-pivot"
WORK="$HOME/mint"

echo "== 0. system deps =="
sudo apt-get update -qq
# libgl1/libglib2.0-0: ai-toolkit imports opencv, which needs libGL.so.1 (absent on headless images)
sudo apt-get install -y -qq git python3-venv python3-pip libgl1 libglib2.0-0 >/dev/null

mkdir -p "$WORK" && cd "$WORK"

echo "== 1. our package =="
if [ ! -d Diffusion-LoRAcle ]; then
  git clone -q --branch "$BRANCH" "$REPO_URL"
fi
cd Diffusion-LoRAcle && git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git pull -q && cd ..

echo "== 2. ai-toolkit =="
if [ ! -d ai-toolkit ]; then
  git clone -q --recursive https://github.com/ostris/ai-toolkit.git
fi

echo "== 3. venv + deps (slow first time) =="
if [ ! -d venv ]; then python3 -m venv venv; fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q --upgrade pip
# torch first so ai-toolkit's requirements resolve against the CUDA build already on the image
# torchaudio too: ai-toolkit's config_modules imports it unconditionally
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 2>/dev/null || \
  pip install -q torch torchvision torchaudio
pip install -q -r ai-toolkit/requirements.txt
pip install -q "diffusers @ git+https://github.com/huggingface/diffusers.git" \
               open_clip_torch safetensors accelerate huggingface_hub
pip install -q -e Diffusion-LoRAcle

echo "== 4. sanity =="
python -c "import torch, diffusers, open_clip; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('diffusers', diffusers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "
Setup complete. Next:
  source $WORK/venv/bin/activate && cd $WORK/Diffusion-LoRAcle
  python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 6
  python scripts/mint_run.py --batch assets/organisms/configs/batch_manifest.json \\
      --plan assets/organisms/mint_plan.json --out assets/organisms/minted_manifest.json --limit 2
"
