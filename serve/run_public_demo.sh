#!/usr/bin/env bash
# run_public_demo.sh - serve the PSA translator from this node on a public URL.
#
#   bash serve/run_public_demo.sh
#
# Starts uvicorn on localhost and opens a Cloudflare quick tunnel to it, which
# gives back a public https://<random>.trycloudflare.com address. Nothing is
# installed system-wide and no Cloudflare account is needed.
#
# WHAT THIS IS NOT: permanent. A quick tunnel's hostname is random and changes
# every time this script runs, and the URL dies when this process or the node
# does. Use it for a live demo. For a link that has to keep working - a link
# in a paper - the service has to outlive this session (see the Hugging Face
# Spaces option instead).
#
# Both checkpoints load from deployment/nllb/{run_id} - this project's own
# layout, and app.py's own default - so on this node it starts in seconds and
# needs no Hugging Face token at all.
#
# Deliberately uses NO curl, wget or git: a Jupyter GPU node often has none of
# them. Python is the one interpreter it's guaranteed to have, so it does the
# downloading and the health-checking too.

set -euo pipefail

PORT="${PORT:-8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ]; then
  echo "no python on PATH - cannot continue"; exit 1
fi

export ENABLED_SYSTEMS="${ENABLED_SYSTEMS:-ekegusii,other_langs}"
export MAX_RESIDENT="${MAX_RESIDENT:-2}"
export PRELOAD="${PRELOAD:-1}"
export RATE_LIMIT_PER_MIN="${RATE_LIMIT_PER_MIN:-30}"

# ---- cloudflared: a single static binary, no install, no root --------------
CF="$HERE/.cloudflared"
if [ ! -x "$CF" ]; then
  echo "downloading cloudflared ..."
  "$PY" - "$CF" <<'PYEOF'
import platform, sys, urllib.request
dest = sys.argv[1]
arch = {"x86_64": "amd64", "amd64": "amd64",
        "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine())
if arch is None:
    sys.exit(f"unsupported architecture: {platform.machine()}")
url = ("https://github.com/cloudflare/cloudflared/releases/latest/download/"
       f"cloudflared-linux-{arch}")
print(f"  {url}")
urllib.request.urlretrieve(url, dest)
PYEOF
  chmod +x "$CF"
  echo "  ok ($(wc -c < "$CF") bytes)"
fi

cleanup() {
  echo
  echo "shutting down ..."
  [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

alive() {
  "$PY" - "$1" <<'PYEOF' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(sys.argv[1], timeout=3).read()
PYEOF
}

# ---- API -------------------------------------------------------------------
# Refuse to start on top of a process already serving this port - otherwise
# the health poll below would succeed against the OLD server, this script
# would report success, and any environment change made since (FEEDBACK_REPO,
# ENABLED_SYSTEMS, RATE_LIMIT_PER_MIN) would silently not apply. That failure
# is invisible and expensive.
if alive "http://127.0.0.1:$PORT/api/health"; then
  echo
  echo "Something is ALREADY serving http://127.0.0.1:$PORT."
  echo "That is almost certainly a previous run of this script that did not die."
  echo "This one would appear to work while actually leaving the old settings in"
  echo "place, so it is stopping instead."
  echo
  echo "Kill it:      pkill -f 'uvicorn app:app' ; pkill -f cloudflared"
  echo "Or move over: PORT=8001 bash serve/run_public_demo.sh"
  exit 1
fi

echo "systems      : $ENABLED_SYSTEMS"
echo "feedback     : ${FEEDBACK_REPO:-(unset - corrections stay on this node and die with it)}"
echo "rate limit   : $RATE_LIMIT_PER_MIN requests/min per IP"
echo "starting the API on 127.0.0.1:$PORT ..."
"$PY" -m uvicorn app:app --host 127.0.0.1 --port "$PORT" > "$HERE/api.log" 2>&1 &
API_PID=$!

for _ in $(seq 1 180); do
  if alive "http://127.0.0.1:$PORT/api/health"; then break; fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "the API died on startup. Last lines of serve/api.log:"
    tail -30 "$HERE/api.log"; exit 1
  fi
  sleep 1
done

if ! alive "http://127.0.0.1:$PORT/api/health"; then
  echo "the API never became healthy. serve/api.log:"; tail -30 "$HERE/api.log"; exit 1
fi
echo "API is up (both models may still be loading in the background; watch serve/api.log)"

# ---- tunnel ------------------------------------------------------------
echo "opening the tunnel ..."
# --protocol http2 avoids the QUIC/UDP 7844 path, which is blocked on a lot of
# university and cloud networks and fails with a confusing timeout.
"$CF" tunnel --no-autoupdate --protocol http2 \
      --url "http://127.0.0.1:$PORT" > "$HERE/tunnel.log" 2>&1 &
TUNNEL_PID=$!

URL=""
for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$HERE/tunnel.log" | head -1 || true)"
  [ -n "$URL" ] && break
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "the tunnel died. Last lines of serve/tunnel.log:"
    tail -30 "$HERE/tunnel.log"; exit 1
  fi
  sleep 1
done

if [ -z "$URL" ]; then
  echo "no URL appeared in 60s. serve/tunnel.log:"; tail -30 "$HERE/tunnel.log"; exit 1
fi

echo "$URL" > "$HERE/PUBLIC_URL.txt"
echo
echo "======================================================================"
echo "  $URL"
echo "======================================================================"
echo "  also written to serve/PUBLIC_URL.txt"
echo
echo "  This address is temporary. It changes every time this script runs and"
echo "  stops working when this process or this node stops."
echo
echo "  Ctrl-C to stop. Logs: serve/api.log, serve/tunnel.log"
echo

wait "$API_PID"
