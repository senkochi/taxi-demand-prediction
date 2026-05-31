#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"

mongo_exec() {
  local container_name="$1"
  local port="$2"
  local eval_code="$3"
  docker exec "${container_name}" mongosh --quiet --port "${port}" --eval "${eval_code}"
}

wait_for_mongo() {
  local container_name="$1"
  local port="$2"
  local retries=60
  local count=0

  until docker exec "${container_name}" mongosh --quiet --port "${port}" --eval "db.adminCommand({ ping: 1 }).ok" >/dev/null 2>&1; do
    count=$((count + 1))
    if [ "${count}" -ge "${retries}" ]; then
      echo "[ERROR] Timeout waiting for ${container_name}"
      exit 1
    fi
    sleep 2
  done

  echo "[OK] ${container_name} is reachable"
}

echo "[INFO] Waiting for MongoDB containers..."
wait_for_mongo "mongo-configsvr" "27019"
wait_for_mongo "mongo-shard1" "27018"
wait_for_mongo "mongo-shard2" "27018"
wait_for_mongo "mongo-shard3" "27018"


echo "[INFO] Initializing config server replica set..."
mongo_exec "mongo-configsvr" "27019" '
try {
  rs.status();
  print("cfgRS already initialized");
} catch (e) {
  rs.initiate({ _id: "cfgRS", configsvr: true, members: [{ _id: 0, host: "configsvr:27019" }] });
  print("cfgRS initialized");
}
'


echo "[INFO] Initializing shard replica sets..."
mongo_exec "mongo-shard1" "27018" '
try {
  rs.status();
  print("shard1RS already initialized");
} catch (e) {
  rs.initiate({ _id: "shard1RS", members: [{ _id: 0, host: "shard1:27018" }] });
  print("shard1RS initialized");
}
'

mongo_exec "mongo-shard2" "27018" '
try {
  rs.status();
  print("shard2RS already initialized");
} catch (e) {
  rs.initiate({ _id: "shard2RS", members: [{ _id: 0, host: "shard2:27018" }] });
  print("shard2RS initialized");
}
'

mongo_exec "mongo-shard3" "27018" '
try {
  rs.status();
  print("shard3RS already initialized");
} catch (e) {
  rs.initiate({ _id: "shard3RS", members: [{ _id: 0, host: "shard3:27018" }] });
  print("shard3RS initialized");
}
'


echo "[INFO] Waiting for replica set primaries..."
sleep 8

wait_for_mongo "mongo-mongos" "27017"


echo "[INFO] Adding shards to mongos..."
mongo_exec "mongo-mongos" "27017" '
const addShardSafe = (name) => {
  try {
    const res = sh.addShard(name);
    printjson(res);
  } catch (e) {
    print("addShard warning: " + e.message);
  }
};
addShardSafe("shard1RS/shard1:27018");
addShardSafe("shard2RS/shard2:27018");
addShardSafe("shard3RS/shard3:27018");
'


echo "[INFO] Enabling sharding and sharding trips collection..."
mongo_exec "mongo-mongos" "27017" '
sh.enableSharding("taxi_db");
db = db.getSiblingDB("taxi_db");
db.createCollection("trips");
try {
  sh.shardCollection("taxi_db.trips", { PULocationID: 1 });
} catch (e) {
  print("shardCollection warning: " + e.message);
}

db.trips.createIndex({ PULocationID: 1 });
db.trips.createIndex({ pickup_datetime: 1 });
db.trips.createIndex({ PULocationID: 1, DOLocationID: 1 });
'


echo "[INFO] Cluster status:"
mongo_exec "mongo-mongos" "27017" 'sh.status()'

echo "[OK] MongoDB sharded cluster is ready"
