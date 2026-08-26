#!/usr/bin/env bash
# Start the fill mint as a system-scope systemd unit on THIS box.
#
# Kept as a script (rather than a long inline ssh command) so the launch is one short, reviewable
# command: `bash ~/launch_fill.sh <shard> <n>`.
#
# System scope + linger is load-bearing: nohup, setsid and `systemd-run --user` all die when the ssh
# session closes, because with Linger=no the per-user systemd manager is torn down on logout and takes
# its children with it (PROGRESS 2026-08-13, failure mode #2).
#
# Usage (on the box):  bash launch_fill.sh <shard_index> <n_shards>
set -uo pipefail
I="${1:?shard index}"; N="${2:?n shards}"

sudo loginctl enable-linger "$USER"
sudo systemctl reset-failed mintfill 2>/dev/null || true
sudo systemd-run --unit=mintfill --collect \
  --uid="$(id -u)" --gid="$(id -g)" \
  --setenv=HOME="$HOME" \
  --working-directory="$HOME" \
  /bin/bash "$HOME/run_fill.sh" "$I" "$N"

sleep 5
echo "mintfill is-active: $(systemctl is-active mintfill)"
