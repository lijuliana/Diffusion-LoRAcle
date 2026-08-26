#!/usr/bin/env bash
# Wait for the mint unit to finish, flush results to the bucket, then STOP THIS INSTANCE.
#
# Why this exists: after the 2026-08-14 gate mint finished, five g2-standard-8 boxes sat RUNNING and
# idle at ~$0.85/hr each until someone noticed nine days later — ~$760 burned for nothing, and the
# adapters were never in the bucket because the sync had been silently 403ing the whole time. A mint
# job must therefore end by doing two things itself: persist its output, and turn the machine off.
# Neither can depend on a human remembering.
#
# Run as its own system-scope unit so it outlives the ssh session that starts it:
#   sudo systemd-run --unit=autostop --collect --setenv=HOME=$HOME /bin/bash $HOME/autostop.sh mintfill
#
# Usage (on the box):  bash autostop.sh <unit-to-watch>
set -uo pipefail
UNIT="${1:?unit to watch}"
REPO="$HOME/mint/Diffusion-LoRAcle"
BUCKET="${BUCKET:-gs://ditloracle-corpus}"
NAME="$(hostname)"
ZONE="$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')"

# PHASE 1 — wait for the unit to actually COME UP before watching it finish.
# Without this the watcher is a footgun: `while systemctl is-active` never enters its loop when the
# unit has not started yet, so the script falls straight through to shutdown. That is not theoretical
# — it terminated ms-5..8 within seconds of launch on 2026-08-24, before a single organism was minted.
for _ in $(seq 1 60); do
  systemctl is-active --quiet "$UNIT" && break
  sleep 20
done

# Never treat "never started" as "finished". A unit that failed to launch is a BUG to diagnose, and
# halting the box would destroy the evidence (journalctl goes with it) and read as a clean completion.
if ! systemctl is-active --quiet "$UNIT"; then
  echo "$(date) — $UNIT never became active after 20 min; NOT shutting down. Investigate with:"
  echo "  sudo journalctl -u $UNIT --no-pager -n 100"
  exit 1
fi
echo "$(date) — $UNIT is up; watching for completion"

# PHASE 2 — watch it run to completion.
while systemctl is-active --quiet "$UNIT"; do sleep 60; done
echo "$(date) — $UNIT finished; flushing before shutdown"

# The compute SA now holds roles/storage.objectAdmin on the bucket (granted 2026-08-23), so this
# finally works unattended; before that it 403'd into /dev/null and the corpus never left the box.
gcloud storage rsync -r "$REPO/assets/organisms/weights/" "$BUCKET/organisms/weights/" 2>&1 | tail -1
for f in "$REPO"/assets/organisms/minted_*.json; do
  [ -e "$f" ] && gcloud storage cp "$f" "$BUCKET/organisms/" 2>&1 | tail -1
done
echo "$(date) — flush complete; stopping $NAME in $ZONE"
# Halt from INSIDE the guest rather than calling the API. Checked 2026-08-24: this project's default
# compute SA has NO project-level IAM binding at all (the same reason the bucket sync 403'd for nine
# days), so `gcloud compute instances stop` would fail here — and an autostop that fails silently is
# worse than none, because it reads as handled. A guest-initiated halt needs no credentials: GCE moves
# the instance to TERMINATED, which ends vCPU/RAM billing (the boot disk still bills, ~$0.04/GB/mo).
# The API call stays as a fallback in case a future box does have the permission and blocks on halt.
sudo shutdown -h now || gcloud compute instances stop "$NAME" --zone "$ZONE" --quiet
