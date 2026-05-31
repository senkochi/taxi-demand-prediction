from pyspark.sql import SparkSession, functions as F
import argparse


def create_spark(app_name="BaselineFeaturesSpark"):
    # Set a local warehouse dir to avoid ViewFileSystem/Hadoop issues on Windows
    return SparkSession.builder.master("local[4]") \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", 200) \
        .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()


def main(input_path: str, output_path: str):
    spark = create_spark()
    df = spark.read.parquet(input_path)

    # 15-minute window aggregation
    windowed = df.withColumn("time_bucket", F.window(F.col("tpep_pickup_datetime"), "15 minutes"))

    agg = windowed.groupBy(F.col("time_bucket"), F.col("PULocationID").alias("zone_id")).agg(
        F.count("*").alias("demand_count"),
        F.avg("fare_amount").alias("avg_fare"),
        F.avg("trip_distance").alias("avg_distance"),
        F.avg("passenger_count").alias("avg_passenger"),
        F.sum("fare_amount").alias("sum_fare"),
        F.expr("percentile_approx(trip_distance, 0.5)").alias("median_distance"),
        F.expr("percentile_approx(trip_distance, 0.95)").alias("p95_distance")
    )

    # flatten window to start/end timestamps and add day/hour
    out = agg.withColumn("window_start", F.col("time_bucket.start")).withColumn("window_end", F.col("time_bucket.end")) \
        .withColumn("date_day", F.dayofweek(F.col("window_start")) - 1) \
        .withColumn("hour", F.hour(F.col("window_start"))) \
        .drop("time_bucket")

    out.write.mode("overwrite").parquet(output_path.rstrip("/") + "/baseline_features_spark/")
    print(f"Saved baseline features to {output_path.rstrip('/')}/baseline_features_spark/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/ingested_data.parquet")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    main(args.input, args.output)
