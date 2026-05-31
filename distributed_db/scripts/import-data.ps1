param(
    [string]$CsvPath = "distributed_db/data/taxi_sample.csv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $CsvPath)) {
    throw "CSV not found: $CsvPath"
}

Write-Host "[INFO] Copying CSV to mongos container"
docker cp $CsvPath mongo-mongos:/tmp/taxi_sample.csv

Write-Host "[INFO] Importing CSV into taxi_db.trips"
docker exec mongo-mongos mongoimport --db taxi_db --collection trips --type csv --headerline --file /tmp/taxi_sample.csv

Write-Host "[INFO] Running validation queries"
docker exec mongo-mongos mongosh --quiet --port 27017 --eval "const dbRef = db.getSiblingDB('taxi_db'); print('total_docs=' + dbRef.trips.countDocuments()); print('sample_doc='); printjson(dbRef.trips.findOne({}, {_id:0}));"

Write-Host "[OK] Import complete"
