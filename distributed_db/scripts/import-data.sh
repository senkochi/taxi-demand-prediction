#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CSV_PATH="${1:-${PROJECT_DIR}/data/taxi_sample.csv}"

if [ ! -f "${CSV_PATH}" ]; then
  echo "[ERROR] CSV not found at ${CSV_PATH}"
  echo "Generate one first, e.g.: python distributed_db/scripts/generate_sample_trips.py --rows 100000"
  exit 1
fi

echo "[INFO] Importing CSV into taxi_db.trips through mongos..."
echo "[INFO] Copying CSV into mongos container path /tmp/taxi_sample.csv"
docker cp "${CSV_PATH}" mongo-mongos:/tmp/taxi_sample.csv

docker exec -i mongo-mongos mongoimport \
  --db taxi_db \
  --collection trips \
  --type csv \
  --headerline \
  --file /tmp/taxi_sample.csv

echo "[INFO] Running validation queries"
docker exec -i mongo-mongos mongosh --quiet --eval '
const dbRef = db.getSiblingDB("taxi_db");
print("total_docs=" + dbRef.trips.countDocuments());
print("sample_doc=");
printjson(dbRef.trips.findOne({}, {_id:0}));
'

echo "[OK] Import complete"
