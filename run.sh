#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

docker compose build
docker compose run --rm detection

echo
echo "Done. Metrics file: results/detection/metrics.json"
