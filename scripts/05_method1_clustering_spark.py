from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import argparse
import os
import pickle


def create_spark(app_name="Method1ClusteringSpark"):
    return SparkSession.builder.master("local[4]") \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", 200) \
        .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()


def run(input_path: str, output_dir: str, k_min=3, k_max=8):
    spark = create_spark()
    df = spark.read.parquet(input_path)

    # hourly demand per zone
    hourly = df.groupBy("PULocationID", F.hour("tpep_pickup_datetime").alias("hour")).agg(F.count("*").alias("demand"))
    pivot = hourly.groupBy("PULocationID").pivot("hour", list(range(24))).agg(F.first("demand")).na.fill(0)
    perf = pivot.withColumnRenamed("PULocationID", "zone_id")

    # compute simple statistics as features
    hour_cols = [str(h) for h in range(24)]
    assembler = VectorAssembler(inputCols=hour_cols, outputCol="raw_features")
    vec = assembler.transform(perf)

    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)
    scaled = scaler.fit(vec).transform(vec)

    evaluator = ClusteringEvaluator(featuresCol="scaled_features", metricName="silhouette")

    best_k, best_score, best_model = None, float("-inf"), None
    for k in range(k_min, k_max + 1):
        km = KMeans(k=k, seed=42, featuresCol="scaled_features", maxIter=100)
        model = km.fit(scaled)
        preds = model.transform(scaled)
        score = evaluator.evaluate(preds)
        print(f"K={k}, silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, model

    assigned = best_model.transform(scaled).select("zone_id", "prediction")
    assigned.write.mode("overwrite").parquet(os.path.join(output_dir, "method1_clusters/assignments/"))
    best_model.write().overwrite().save(os.path.join(output_dir, "method1_clusters/model"))

    # export pickle artifact for training compatibility
    assignments = {row[0]: int(row[1]) for row in assigned.collect()}
    centers = [list(map(float, c.toArray())) for c in best_model.clusterCenters()]
    artifact = {
        "zone_to_cluster": assignments,
        "silhouette_score": float(best_score),
        "optimal_k": int(best_k),
        "cluster_centers": centers,
    }
    os.makedirs("data/models", exist_ok=True)
    with open(f"data/models/method1_clusters.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved Method1 clusters (k={best_k}) and artifact to data/models/method1_clusters.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/baseline_features_spark/")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    run(args.input, args.output)
