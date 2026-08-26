#!/usr/bin/env bash
# Chain one mint unit into the next ON THE BOX, so the handover cannot be lost with the operator's
# session. The first attempt ran this wait-loop on a laptop; the loop was killed with the session and
# the box would have gone on to halt itself, silently dropping its shard of the corpus. Anything that
# must outlive an ssh connection belongs in systemd on the machine, not in a terminal.
#
# Usage (on the box):  handover.sh <unit-to-wait-for> <shard_index> <n_shards>
set -uo pipefail
WAIT_UNIT="${1:?unit to wait for}"; I="${2:?shard}"; N="${3:?n shards}"

while systemctl is-active --quiet "$WAIT_UNIT"; do sleep 60; done
echo "$(date) — $WAIT_UNIT finished; starting workshop shard $I/$N"

sudo systemctl reset-failed mintwork autostop 2>/dev/null || true
sudo systemd-run --unit=mintwork --collect --uid="$(id -u)" --gid="$(id -g)" \
  --setenv=HOME="$HOME" --working-directory="$HOME" /bin/bash "$HOME/run_workshop.sh" "$I" "$N"

# Arm the watcher only after confirming the mint is up. Arming it first is what terminated four boxes
# earlier today: the watcher saw a not-yet-active unit, treated that as "finished", and halted.
sleep 25
if systemctl is-active --quiet mintwork; then
  sudo systemd-run --unit=autostop --collect --setenv=HOME="$HOME" /bin/bash "$HOME/autostop.sh" mintwork
  echo "$(date) — mintwork up, autostop armed"
else
  echo "$(date) — mintwork FAILED to start; leaving box up for diagnosis" >&2
  exit 1
fi
