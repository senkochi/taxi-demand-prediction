const dbRef = db.getSiblingDB("taxi_db");

const pipeline = [
  {
    $group: {
      _id: { PULocationID: "$PULocationID", DOLocationID: "$DOLocationID" },
      trip_count: { $sum: 1 },
    },
  },
  { $sort: { trip_count: -1 } },
  { $limit: 5 },
];

printjson(dbRef.trips.aggregate(pipeline).toArray());
