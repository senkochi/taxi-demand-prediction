#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUERIES_FILE="${SCRIPT_DIR}/queries.js"

if [ ! -f "${QUERIES_FILE}" ]; then
  echo "[ERROR] queries.js not found"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -q '^mongo-mongos$'; then
  echo "[ERROR] mongo-mongos container is not running"
  exit 1
fi

echo "[INFO] Copying query script to mongos and executing"
docker cp "${QUERIES_FILE}" mongo-mongos:/tmp/queries.js

docker exec -i mongo-mongos mongosh --quiet /tmp/queries.js

echo "[OK] Distributed queries executed"
