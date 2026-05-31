const addShardSafe = (name) => {
  try {
    printjson(sh.addShard(name));
  } catch (e) {
    print("addShard warning: " + e.message);
  }
};

addShardSafe("shard1RS/shard1:27018");
addShardSafe("shard2RS/shard2:27018");
addShardSafe("shard3RS/shard3:27018");

sh.enableSharding("taxi_db");
db = db.getSiblingDB("taxi_db");

if (!db.getCollectionNames().includes("trips")) {
  db.createCollection("trips");
}

try {
  sh.shardCollection("taxi_db.trips", { PULocationID: 1 });
} catch (e) {
  print("shardCollection warning: " + e.message);
}

db.trips.createIndex({ PULocationID: 1 });
db.trips.createIndex({ pickup_datetime: 1 });
db.trips.createIndex({ PULocationID: 1, DOLocationID: 1 });

sh.status();
