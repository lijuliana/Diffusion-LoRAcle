#!/usr/bin/env bash
# Fan a mint run across N preemptible L4 boxes, then collect the shard manifests.
#
# Wall-clock is the constraint, not cost: one L4 mints ~1.5 organisms/hour, so the 47-organism gate
# set is ~30h serially and ~4h across 8 boxes. Preemptible L4s are ~$0.22/hr, so a full gate mint is
# a couple of dollars either way; `mint_run` is resumable and shards round-robin, so a preemption
# costs one organism and a slice of each axis rather than a whole axis.
#
# Usage:  bash scripts/cluster/fanout_mint.sh <n_boxes> <split> [n_images]
#   bash scripts/cluster/fanout_mint.sh 8 gate 12
set -euo pipefail

N="${1:?usage: fanout_mint.sh <n_boxes> <split> [n_images]}"
SPLIT="${2:?}"
NIMG="${3:-12}"

PROJECT="project-ca10b891-b467-44b6-b56"
ACCOUNT="25julianal@gmail.com"
ZONE="us-central1-a"
BUCKET="gs://ditloracle-corpus"
G="gcloud --project=$PROJECT --account=$ACCOUNT"

echo "== creating $N preemptible L4 boxes =="
for i in $(seq 0 $((N-1))); do
  name="ditloracle-mint-$i"
  $G compute instances describe "$name" --zone "$ZONE" >/dev/null 2>&1 && { echo "  $name exists"; continue; }
  $G compute instances create "$name" --zone="$ZONE" --machine-type=g2-standard-8 \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
    --boot-disk-size=300GB --boot-disk-type=pd-balanced \
    --provisioning-model=SPOT --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform >/dev/null &
done
wait
echo "  created."

echo "== setup + launch shard on each box =="
for i in $(seq 0 $((N-1))); do
  name="ditloracle-mint-$i"
  (
    until $G compute ssh "$name" --zone "$ZONE" --quiet --command "true" >/dev/null 2>&1; do sleep 15; done
    $G compute scp scripts/cluster/setup_mint_box.sh "$name":~/setup_mint_box.sh --zone "$ZONE" --quiet >/dev/null
    $G compute ssh "$name" --zone "$ZONE" --quiet --command "
      bash ~/setup_mint_box.sh > ~/setup.log 2>&1
      cd ~/mint/Diffusion-LoRAcle && source ~/mint/venv/bin/activate && export PYTHONPATH=.
      python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 6 --n-images $NIMG >/dev/null
      nohup python -u scripts/mint_run.py \
        --batch assets/organisms/configs/batch_manifest.json \
        --plan assets/organisms/mint_plan.json \
        --out assets/organisms/minted_${SPLIT}_shard$i.json \
        --aitk-dir ~/mint/ai-toolkit --split $SPLIT --shard $i/$N --n-images $NIMG \
        > ~/mint/mint_shard.log 2>&1 &
      echo 'shard $i launched'
    " 2>&1 | tail -1
  ) &
done
wait

cat <<EOF

All $N shards launched. Watch one:
  gcloud compute ssh ditloracle-mint-0 --zone $ZONE --project $PROJECT --command 'tail -f ~/mint/mint_shard.log'

When done, collect weights + manifests to the bucket, then merge:
  for i in \$(seq 0 $((N-1))); do
    gcloud compute ssh ditloracle-mint-\$i --zone $ZONE --project $PROJECT --command \\
      "gsutil -m cp ~/mint/Diffusion-LoRAcle/assets/organisms/weights/*.safetensors $BUCKET/organisms/weights/ ; \\
       gsutil cp ~/mint/Diffusion-LoRAcle/assets/organisms/minted_${SPLIT}_shard\$i.json $BUCKET/organisms/"
  done
  python scripts/merge_minted.py --bucket $BUCKET --split $SPLIT --out assets/organisms/minted_${SPLIT}.json

Delete the boxes when finished (they are SPOT, but still billed while running):
  for i in \$(seq 0 $((N-1))); do gcloud compute instances delete ditloracle-mint-\$i --zone $ZONE --project $PROJECT --quiet & done; wait
EOF
