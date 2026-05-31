const dbRef = db.getSiblingDB("taxi_db");

const pipeline = [
  {
    $project: {
      PULocationID: 1,
      pickup_hour: {
        $dateToString: {
          format: "%Y-%m-%dT%H:00:00",
          date: { $toDate: "$pickup_datetime" },
        },
      },
    },
  },
  {
    $group: {
      _id: { PULocationID: "$PULocationID", pickup_hour: "$pickup_hour" },
      trip_count: { $sum: 1 },
    },
  },
  { $sort: { trip_count: -1 } },
  { $limit: 5 },
];

printjson(dbRef.trips.aggregate(pipeline).toArray());
