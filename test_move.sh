#!/usr/bin/env bash
set -euo pipefail

PORT="${SERVE_PORT:-5058}"
URL="http://127.0.0.1:${PORT}/move"

# You can change this PGN to whatever line you want to test
PGN="${1:-"1. e4"}"
TIMELEFT_MS="${2:-60000}"

JSON_PAYLOAD=$(cat <<JSON
{
  "pgn": "$PGN",
  "timeleft": $TIMELEFT_MS
}
JSON
)

echo "=== Testing /move ==="
echo "POST $URL"
echo "Request JSON:"
echo "$JSON_PAYLOAD"
echo

curl -v -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$JSON_PAYLOAD"

echo
echo "=== Done ==="
