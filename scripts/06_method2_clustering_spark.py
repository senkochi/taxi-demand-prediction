from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import argparse
import os
import pickle


def create_spark(app_name="Method2ClusteringSpark"):
    return SparkSession.builder.master("local[4]") \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", 200) \
        .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()


def run(baseline_path: str, output_dir: str, k_min=3, k_max=8):
    spark = create_spark()
    bf = spark.read.parquet(baseline_path)

    mobility = bf.groupBy(F.col("zone_id").alias("zone_id")).agg(
        F.sum("demand_count").alias("trip_count"),
        F.avg("avg_distance").alias("avg_trip_distance"),
        F.avg("avg_fare").alias("avg_fare"),
        F.avg("avg_passenger").alias("avg_passenger_count"),
        F.avg((F.col("window_end").cast("long") - F.col("window_start").cast("long")) / 60.0).alias("avg_trip_duration_min")
    ).na.fill(0)

    cols = ["trip_count", "avg_trip_distance", "avg_fare", "avg_passenger_count", "avg_trip_duration_min"]
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    vec = assembler.transform(mobility)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    scaled = scaler.fit(vec).transform(vec)

    evaluator = ClusteringEvaluator(featuresCol="scaled_features", metricName="silhouette")

    best_k = None
    best_score = float("-inf")
    best_model = None

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(k=k, seed=42, featuresCol="scaled_features", maxIter=100)
        model = kmeans.fit(scaled)
        preds = model.transform(scaled)
        score = evaluator.evaluate(preds)
        print(f"K={k}, silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    final = best_model.transform(scaled).select("zone_id", "prediction")
    final.write.mode("overwrite").parquet(os.path.join(output_dir, "method2_clusters/assignments/"))
    best_model.write().overwrite().save(os.path.join(output_dir, "method2_clusters/model"))

    # Export pickle artifact
    assignments = {row[0]: int(row[1]) for row in final.collect()}
    centers = [list(map(float, c.toArray())) for c in best_model.clusterCenters()]
    artifact = {
        "zone_to_cluster": assignments,
        "silhouette_score": float(best_score),
        "optimal_k": int(best_k),
        "cluster_centers": centers,
    }
    os.makedirs("data/models", exist_ok=True)
    with open(f"data/models/method2_clusters.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved Method2 clustering (k={best_k}) and artifact to data/models/method2_clusters.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/processed/baseline_features_spark/")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    run(args.baseline, args.output)
