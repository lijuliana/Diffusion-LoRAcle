#!/usr/bin/env bash
# Push finished organisms to the bucket every few minutes, for the whole life of a mint.
#
# Needed because the workshop run is ~35 h and half the fleet is spot. Two boxes were preempted within
# the first hour. Their disks survive (the instances use --instance-termination-action=STOP, not
# DELETE, after a 2026-08-13 preemption deleted a box along with its finished adapters), but a box that
# only flushes at the END strands every organism it has produced for as long as it stays down, and the
# merge cannot see them. Syncing continuously makes a preemption cost the organism in flight and
# nothing else.
#
# This is the loop that already existed in run_shard.sh and silently 403'd for nine days because the
# compute SA had no bucket permission. It works now (objectAdmin granted 2026-08-23), and it no longer
# hides its errors.
#
# Usage (on the box):  sudo systemd-run --unit=syncloop --collect --setenv=HOME=$HOME \
#                        /bin/bash $HOME/syncloop.sh
set -uo pipefail
REPO="$HOME/mint/Diffusion-LoRAcle"
BUCKET="${BUCKET:-gs://ditloracle-corpus}"

while true; do
  if ! gcloud storage rsync -r "$REPO/assets/organisms/weights/" "$BUCKET/organisms/weights/" 2>&1 | tail -1; then
    echo "$(date) — WARNING: sync failed (check bucket IAM)" >&2
  fi
  for f in "$REPO"/assets/organisms/minted_*.json; do
    [ -e "$f" ] && gcloud storage cp "$f" "$BUCKET/organisms/" 2>&1 | tail -1
  done
  sleep 300
done
