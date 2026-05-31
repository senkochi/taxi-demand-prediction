// Methodology-only benchmark helper.
// This script captures timings and explain plans but does not claim any results.

const dbRef = db.getSiblingDB("taxi_db");

function runTimed(label, fn) {
  const t0 = Date.now();
  const output = fn();
  const t1 = Date.now();
  print(`${label}_ms=${t1 - t0}`);
  return output;
}

const demandPipeline = [
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
      _id: { PULocationID: "$PULocationID", pickup_hour: "$pickup_hour" },
      trip_count: { $sum: 1 },
      avg_trip_distance: { $avg: "$trip_distance" },
    },
  },
];

const odPipeline = [
  {
    $group: {
      _id: { PULocationID: "$PULocationID", DOLocationID: "$DOLocationID" },
      trip_count: { $sum: 1 },
    },
  },
  { $sort: { trip_count: -1 } },
];

print("total_docs=" + dbRef.trips.countDocuments());

runTimed("demand_agg", () => dbRef.trips.aggregate(demandPipeline).toArray());
runTimed("od_agg", () => dbRef.trips.aggregate(odPipeline).toArray());

print("demand_explain=");
printjson(dbRef.trips.explain("executionStats").aggregate(demandPipeline));

print("od_explain=");
printjson(dbRef.trips.explain("executionStats").aggregate(odPipeline));
