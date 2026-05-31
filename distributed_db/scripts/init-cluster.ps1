$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Wait-Mongo($ContainerName, $Port, $Retries = 60) {
    for ($i = 0; $i -lt $Retries; $i++) {
    docker exec $ContainerName mongosh --quiet --port $Port --eval "db.adminCommand({ ping: 1 }).ok" *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] $ContainerName is reachable"
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Timeout waiting for $ContainerName"
}

Write-Host "[INFO] Waiting for MongoDB containers..."
Wait-Mongo "mongo-configsvr" 27019
Wait-Mongo "mongo-shard1" 27018
Wait-Mongo "mongo-shard2" 27018
Wait-Mongo "mongo-shard3" 27018

Write-Host "[INFO] Copying init scripts into containers"
docker cp "$ScriptDir/init-configsvr.js" mongo-configsvr:/tmp/init-configsvr.js
docker cp "$ScriptDir/init-shard1.js" mongo-shard1:/tmp/init-shard1.js
docker cp "$ScriptDir/init-shard2.js" mongo-shard2:/tmp/init-shard2.js
docker cp "$ScriptDir/init-shard3.js" mongo-shard3:/tmp/init-shard3.js
docker cp "$ScriptDir/init-mongos.js" mongo-mongos:/tmp/init-mongos.js

Write-Host "[INFO] Initializing config server replica set"
docker exec mongo-configsvr mongosh --quiet --port 27019 /tmp/init-configsvr.js

Write-Host "[INFO] Initializing shard replica sets"
docker exec mongo-shard1 mongosh --quiet --port 27018 /tmp/init-shard1.js
docker exec mongo-shard2 mongosh --quiet --port 27018 /tmp/init-shard2.js
docker exec mongo-shard3 mongosh --quiet --port 27018 /tmp/init-shard3.js

Start-Sleep -Seconds 8

Wait-Mongo "mongo-mongos" 27017

Write-Host "[INFO] Adding shards and enabling sharding"
docker exec mongo-mongos mongosh --quiet --port 27017 /tmp/init-mongos.js

Write-Host "[INFO] Cluster status"
docker exec mongo-mongos mongosh --quiet --port 27017 --eval "sh.status()"

Write-Host "[OK] MongoDB sharded cluster is ready"
