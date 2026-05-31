const ns = "taxi_db.trips";

print("before=");
printjson(sh.status());

// Force 3 chunks and move them to different shards.
sh.splitAt(ns, { PULocationID: 90 });
sh.splitAt(ns, { PULocationID: 180 });

sh.moveChunk(ns, { PULocationID: 10 }, "shard1RS");
sh.moveChunk(ns, { PULocationID: 100 }, "shard2RS");
sh.moveChunk(ns, { PULocationID: 220 }, "shard3RS");

print("after=");
printjson(sh.status());
