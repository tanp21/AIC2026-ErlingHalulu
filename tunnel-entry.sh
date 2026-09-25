#!/bin/sh
# Hold the venue link to the compute host open, reconnecting forever.
#
# This runs inside the docker network rather than on the host on purpose. A host-side tunnel
# would have to bind 0.0.0.0 for the app container to reach it (a container's 127.0.0.1 is its
# own), and that would put the compute host on the venue LAN -- where anyone could present any
# X-AIC-User header, which is the whole of its authentication.
set -u

: "${AIC_CORE_SSH_HOST:?set AIC_CORE_SSH_HOST to the compute host}"
: "${AIC_CORE_SSH_KEY:?set AIC_CORE_SSH_KEY to the private key mounted into this container}"

PORT="${AIC_CORE_SSH_PORT:-22}"
USER="${AIC_CORE_SSH_USER:-root}"
LOCAL_PORT="${AIC_TUNNEL_PORT:-9000}"

# ssh refuses to use a key anybody but its owner can read, and a bind-mounted key keeps the
# host's mode. Copy it somewhere writable with the mode ssh wants.
KEY=/tmp/identity
cp "$AIC_CORE_SSH_KEY" "$KEY" && chmod 600 "$KEY"

while true; do
  ssh -N \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=accept-new \
      -o UserKnownHostsFile=/tmp/known_hosts \
      -o BatchMode=yes \
      -i "$KEY" -p "$PORT" \
      -L "0.0.0.0:${LOCAL_PORT}:127.0.0.1:8000" \
      "${USER}@${AIC_CORE_SSH_HOST}"
  echo "tunnel dropped at $(date '+%F %T'), reconnecting in 5s"
  sleep 5
done
