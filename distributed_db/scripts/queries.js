const dbRef = db.getSiblingDB("taxi_db");

// 1) Demand Statistics: grouped by PULocationID and pickup hour
const demandStatsPipeline = [
  {
    $addFields: {
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
      _id: {
        PULocationID: "$PULocationID",
        pickup_hour: "$pickup_hour",
      },
      trip_count: { $sum: 1 },
      avg_trip_distance: { $avg: "$trip_distance" },
      avg_passenger_count: { $avg: "$passenger_count" },
    },
  },
  { $sort: { "_id.pickup_hour": 1, "_id.PULocationID": 1 } },
  { $limit: 100 },
];

print("Demand statistics preview:");
printjson(dbRef.trips.aggregate(demandStatsPipeline).toArray());

// 2) OD Statistics: grouped by origin-destination pair
const odStatsPipeline = [
  {
    $group: {
      _id: {
        PULocationID: "$PULocationID",
        DOLocationID: "$DOLocationID",
      },
      trip_count: { $sum: 1 },
      avg_trip_distance: { $avg: "$trip_distance" },
    },
  },
  { $sort: { trip_count: -1 } },
  { $limit: 100 },
];

print("OD statistics preview:");
printjson(dbRef.trips.aggregate(odStatsPipeline).toArray());
