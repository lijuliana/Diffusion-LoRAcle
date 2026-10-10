#!/usr/bin/env bash
# One-time teardown (2026-10-10): delete the 3 idle disk-inspector VMs and every terminated
# ditloracle fleet instance plus their disks (~$1,057/mo). Everything of value is verified in
# gs://ditloracle-corpus (831 adapter weights, sweep8 results + reader checkpoints, h100 salvage,
# all sweep JSONs/logs). Leaves cs2881r-olmo-rollouts (other course project) untouched.
set -uo pipefail
P=project-ca10b891-b467-44b6-b56

echo "== deleting instances =="
gcloud compute instances delete disk-inspector \
  ditloracle-a100-{1..8} ditloracle-h100-1 ditloracle-mint-l4 \
  ditloracle-ms-{5,6,9,10,11,12,13,14,15,16,17,18} ditloracle-reader-a100 \
  --zone=us-central1-a --project=$P --quiet
gcloud compute instances delete disk-inspector-b ditloracle-ms-7 ditloracle-ms-8 \
  --zone=us-central1-b --project=$P --quiet
gcloud compute instances delete disk-inspector-w ditloracle-ms-{1..4} \
  --zone=us-west1-a --project=$P --quiet

echo "== deleting any disks the instance deletions left behind =="
for z in us-central1-a us-central1-b us-west1-a; do
  leftover=$(gcloud compute disks list --project=$P \
    --filter="zone:$z AND (name~^ditloracle- OR name~^disk-inspector)" \
    --format="value(name)")
  [ -n "$leftover" ] && gcloud compute disks delete $leftover --zone=$z --project=$P --quiet
done

echo "== remaining instances and disks =="
gcloud compute instances list --project=$P
gcloud compute disks list --project=$P --format="table(name,sizeGb,type.basename())"
