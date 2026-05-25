"""
Week 1.2: Data Validation & Quality Checks Pipeline
Purpose: Validate ingested data, detect outliers, and produce cleaned dataset

Input: Parquet file from scripts/01_data_ingestion.py
Output: 
  - data/processed/validated_data.parquet (cleaned, quality-assured)
  - logs/02_validation_report.json (comprehensive quality report)

Execution: python scripts/02_data_validation.py

Validation Checks:
  1. Schema validation (17+ required columns)
  2. Null value handling
  3. Outlier detection (Z-score method)
  4. Date range validation
  5. Location ID bounds checking
  6. Data type consistency
"""

import sys
import os
import json
import logging
import io
from datetime import datetime
from typing import Dict, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, abs, mean, stddev, count, when, 
    min as spark_min, max as spark_max, 
    isnull, isnan
)

# Setup logging with UTF-8 support for Windows
class UTF8StreamHandler(logging.StreamHandler):
    """Stream handler with UTF-8 encoding support for Windows"""
    def __init__(self):
        super().__init__()
        self.stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/02_validation.log', encoding='utf-8'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation and cleaning"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.validation_report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "statistics": {},
            "warnings": [],
            "errors": []
        }
    
    def load_ingested_data(self, input_path: str) -> DataFrame:
        """Load ingested Parquet data"""
        logger.info(f"[DIR] Loading ingested data from {input_path}...")
        df = self.spark.read.parquet(input_path)
        logger.info(f"[OK] Loaded {df.count():,} records")
        return df
    
    def check_schema(self, df: DataFrame) -> Tuple[bool, Dict]:
        """
        Validate schema: ensure all required columns exist
        
        Required: 17+ columns including:
        - tpep_pickup_datetime (REQUIRED)
        - tpep_dropoff_datetime (REQUIRED)
        - PULocationID (REQUIRED)
        - DOLocationID (REQUIRED)
        - fare_amount (REQUIRED)
        - passenger_count (REQUIRED)
        - trip_distance (REQUIRED)
        """
        logger.info("🔍 Validating schema...")
        
        required_cols = {
            'tpep_pickup_datetime': 'timestamp',
            'tpep_dropoff_datetime': 'timestamp',
            'PULocationID': 'integer',
            'DOLocationID': 'integer',
            'fare_amount': 'double',
            'passenger_count': 'integer',
            'trip_distance': 'double',
            'payment_type': 'integer',
            'total_amount': 'double'
        }
        
        df_cols = {field.name: str(field.dataType) for field in df.schema.fields}
        
        check_result = {
            "total_columns": len(df.columns),
            "required_columns_found": 0,
            "missing_columns": [],
            "all_present": True
        }
        
        for req_col, req_type in required_cols.items():
            if req_col in df_cols:
                check_result["required_columns_found"] += 1
            else:
                check_result["missing_columns"].append(req_col)
                check_result["all_present"] = False
        
        if check_result["all_present"]:
            logger.info(f"[OK] Schema valid: {check_result['total_columns']} columns, all required fields present")
        else:
            logger.warning(f"⚠️  Missing columns: {check_result['missing_columns']}")
        
        return check_result["all_present"], check_result
    
    def check_nulls(self, df: DataFrame, tolerance=0.05) -> Tuple[DataFrame, Dict]:
        """
        Handle null values
        
        Strategy: Drop rows with nulls in required fields
        Tolerance: Allow up to 5% nulls in optional fields
        
        Args:
            df: Input DataFrame
            tolerance: Allowed null percentage (default 5%)
            
        Returns:
            (cleaned_df, null_statistics)
        """
        logger.info("🔍 Checking null values...")
        
        required_fields = ['tpep_pickup_datetime', 'PULocationID', 'DOLocationID', 'fare_amount']
        
        null_stats = {
            "before_count": df.count(),
            "null_counts": {},
            "null_percentages": {},
            "rows_with_any_null": 0
        }
        
        # Count nulls per column
        for col_name in df.columns:
            null_count = df.filter(col(col_name).isNull()).count()
            null_pct = (null_count / null_stats["before_count"]) * 100 if null_stats["before_count"] > 0 else 0
            
            null_stats["null_counts"][col_name] = int(null_count)
            null_stats["null_percentages"][col_name] = round(null_pct, 2)
            
            if null_pct > 0:
                logger.info(f"  - {col_name}: {null_count:,} nulls ({null_pct:.2f}%)")
        
        # Drop rows with null in required fields
        df_cleaned = df
        for req_field in required_fields:
            df_cleaned = df_cleaned.filter(col(req_field).isNotNull())
        
        null_stats["after_count"] = df_cleaned.count()
        null_stats["rows_removed"] = null_stats["before_count"] - null_stats["after_count"]
        null_stats["retention_rate"] = round(
            (null_stats["after_count"] / null_stats["before_count"]) * 100, 2
        ) if null_stats["before_count"] > 0 else 100.0
        
        logger.info(f"✅ Null handling: {null_stats['rows_removed']:,} rows removed, "
                   f"{null_stats['retention_rate']:.1f}% retained")
        
        return df_cleaned, null_stats
    
    def check_date_range(self, df: DataFrame, min_date="2019-01-01", max_date="2020-06-30") -> Tuple[DataFrame, Dict]:
        """
        Validate date range
        
        Note: Data is from 01-2019 to 06-2020, but validation range can be customized.
        Current config: 2019-01-01 to 2020-06-30
        
        Args:
            df: Input DataFrame
            min_date: Minimum expected date
            max_date: Maximum expected date
            
        Returns:
            (filtered_df, date_statistics)
        """
        logger.info(f"🔍 Validating date range ({min_date} to {max_date})...")
        
        date_stats = {
            "before_count": df.count(),
            "min_date_in_data": None,
            "max_date_in_data": None,
            "out_of_range_count": 0,
            "retention_rate": 100.0
        }
        
        # Get actual date range
        date_range = df.agg(
            spark_min("tpep_pickup_datetime").alias("min_date"),
            spark_max("tpep_pickup_datetime").alias("max_date")
        ).collect()[0]
        
        if date_range[0]:
            date_stats["min_date_in_data"] = str(date_range[0])
        if date_range[1]:
            date_stats["max_date_in_data"] = str(date_range[1])
        
        logger.info(f"  Data date range: {date_stats['min_date_in_data']} to {date_stats['max_date_in_data']}")
        
        # Filter by expected range
        df_filtered = df.filter(
            (col("tpep_pickup_datetime") >= min_date) &
            (col("tpep_pickup_datetime") <= max_date)
        )
        
        date_stats["out_of_range_count"] = date_stats["before_count"] - df_filtered.count()
        date_stats["retention_rate"] = round(
            (df_filtered.count() / date_stats["before_count"]) * 100, 2
        ) if date_stats["before_count"] > 0 else 100.0
        
        if date_stats["out_of_range_count"] > 0:
            logger.warning(f"⚠️  {date_stats['out_of_range_count']:,} records outside date range removed")
        else:
            logger.info(f"✅ All records within date range")
        
        return df_filtered, date_stats
    
    def detect_outliers(self, df: DataFrame, z_threshold=3.0) -> Tuple[DataFrame, Dict]:
        """
        Detect and remove outliers using Z-score method
        
        Z-score = |value - mean| / stddev
        Remove rows where |Z| > threshold (default 3.0)
        
        Applied to:
        - fare_amount
        - trip_distance
        - passenger_count
        
        Args:
            df: Input DataFrame
            z_threshold: Z-score threshold (default 3.0 = remove 0.3% extreme values)
            
        Returns:
            (cleaned_df, outlier_statistics)
        """
        logger.info(f"🔍 Detecting outliers (Z-score > {z_threshold})...")
        
        outlier_cols = ['fare_amount', 'trip_distance', 'passenger_count']
        outlier_stats = {
            "before_count": df.count(),
            "outliers_by_column": {},
            "total_outliers": 0,
            "retention_rate": 100.0
        }
        
        df_cleaned = df
        
        for col_name in outlier_cols:
            # Calculate mean and stddev
            stats = df_cleaned.agg(
                mean(col_name).alias("mean_val"),
                stddev(col_name).alias("std_val")
            ).collect()[0]
            
            mean_val = stats[0]
            std_val = stats[1]
            
            if std_val is None or std_val == 0:
                logger.warning(f"  ⚠️  {col_name}: Cannot compute stddev (might be constant)")
                outlier_stats["outliers_by_column"][col_name] = 0
                continue
            
            # Filter outliers
            before_count = df_cleaned.count()
            df_cleaned = df_cleaned.filter(
                (abs(col(col_name) - mean_val) / std_val) <= z_threshold
            )
            after_count = df_cleaned.count()
            outliers_removed = before_count - after_count
            
            outlier_stats["outliers_by_column"][col_name] = int(outliers_removed)
            outlier_stats["total_outliers"] += outliers_removed
            
            logger.info(f"  - {col_name}: {outliers_removed:,} outliers removed "
                       f"(μ={mean_val:.2f}, σ={std_val:.2f})")
        
        outlier_stats["after_count"] = df_cleaned.count()
        outlier_stats["retention_rate"] = round(
            (df_cleaned.count() / outlier_stats["before_count"]) * 100, 2
        ) if outlier_stats["before_count"] > 0 else 100.0
        
        logger.info(f"✅ Outlier detection: {outlier_stats['total_outliers']:,} outliers removed, "
                   f"{outlier_stats['retention_rate']:.1f}% retained")
        
        return df_cleaned, outlier_stats
    
    def check_location_ids(self, df: DataFrame, valid_range=(1, 263)) -> Tuple[DataFrame, Dict]:
        """
        Validate LocationID bounds
        
        NYC zones: 1-263 (per taxi_zone_lookup.csv)
        
        Args:
            df: Input DataFrame
            valid_range: Valid location ID range (min, max)
            
        Returns:
            (filtered_df, location_statistics)
        """
        logger.info(f"🔍 Validating LocationIDs (valid range: {valid_range[0]}-{valid_range[1]})...")
        
        loc_stats = {
            "before_count": df.count(),
            "invalid_pu_count": 0,
            "invalid_do_count": 0,
            "retention_rate": 100.0
        }
        
        min_loc, max_loc = valid_range
        
        # Check PULocationID
        invalid_pu = df.filter(
            (col("PULocationID") < min_loc) | (col("PULocationID") > max_loc)
        ).count()
        loc_stats["invalid_pu_count"] = invalid_pu
        
        # Check DOLocationID
        invalid_do = df.filter(
            (col("DOLocationID") < min_loc) | (col("DOLocationID") > max_loc)
        ).count()
        loc_stats["invalid_do_count"] = invalid_do
        
        # Filter valid locations
        df_filtered = df.filter(
            (col("PULocationID") >= min_loc) & (col("PULocationID") <= max_loc) &
            (col("DOLocationID") >= min_loc) & (col("DOLocationID") <= max_loc)
        )
        
        loc_stats["after_count"] = df_filtered.count()
        loc_stats["retention_rate"] = round(
            (df_filtered.count() / loc_stats["before_count"]) * 100, 2
        ) if loc_stats["before_count"] > 0 else 100.0
        
        if invalid_pu + invalid_do > 0:
            logger.warning(f"⚠️  Invalid LocationIDs: {invalid_pu} PU, {invalid_do} DO")
        else:
            logger.info(f"✅ All LocationIDs valid")
        
        return df_filtered, loc_stats
    
    def compute_statistics(self, df: DataFrame) -> Dict:
        """Compute final statistics"""
        logger.info("📊 Computing final statistics...")
        
        stats = {
            "record_count": df.count(),
            "column_count": len(df.columns),
            "fare_stats": {},
            "distance_stats": {},
            "passenger_stats": {}
        }
        
        # Fare statistics
        fare_agg = df.agg(
            mean("fare_amount").alias("mean"),
            stddev("fare_amount").alias("stddev"),
            spark_min("fare_amount").alias("min"),
            spark_max("fare_amount").alias("max")
        ).collect()[0]
        
        stats["fare_stats"] = {
            "mean": round(fare_agg[0], 2) if fare_agg[0] else None,
            "stddev": round(fare_agg[1], 2) if fare_agg[1] else None,
            "min": round(fare_agg[2], 2) if fare_agg[2] else None,
            "max": round(fare_agg[3], 2) if fare_agg[3] else None
        }
        
        # Distance statistics
        dist_agg = df.agg(
            mean("trip_distance").alias("mean"),
            stddev("trip_distance").alias("stddev"),
            spark_min("trip_distance").alias("min"),
            spark_max("trip_distance").alias("max")
        ).collect()[0]
        
        stats["distance_stats"] = {
            "mean": round(dist_agg[0], 2) if dist_agg[0] else None,
            "stddev": round(dist_agg[1], 2) if dist_agg[1] else None,
            "min": round(dist_agg[2], 2) if dist_agg[2] else None,
            "max": round(dist_agg[3], 2) if dist_agg[3] else None
        }
        
        # Passenger statistics
        pass_agg = df.agg(
            mean("passenger_count").alias("mean"),
            spark_min("passenger_count").alias("min"),
            spark_max("passenger_count").alias("max")
        ).collect()[0]
        
        stats["passenger_stats"] = {
            "mean": round(pass_agg[0], 2) if pass_agg[0] else None,
            "min": int(pass_agg[1]) if pass_agg[1] else None,
            "max": int(pass_agg[2]) if pass_agg[2] else None
        }
        
        return stats
    
    def validate_and_clean(self, input_path: str, output_path: str) -> int:
        """
        Execute complete validation pipeline
        
        Returns:
            0 if successful, 1 if errors
        """
        logger.info("=" * 70)
        logger.info("🚕 NYC TAXI DATA VALIDATION PIPELINE (Week 1.2 - Step 2)")
        logger.info("=" * 70)
        
        try:
            # Load data
            df = self.load_ingested_data(input_path)
            original_count = df.count()
            
            # Check 1: Schema validation
            schema_ok, schema_check = self.check_schema(df)
            self.validation_report["checks"]["schema"] = schema_check
            
            if not schema_ok:
                logger.error("❌ Schema validation failed!")
                return 1
            
            # Check 2: Null handling
            df, null_stats = self.check_nulls(df)
            self.validation_report["checks"]["nulls"] = null_stats
            
            # Check 3: Date range validation
            df, date_stats = self.check_date_range(df)
            self.validation_report["checks"]["date_range"] = date_stats
            
            # Check 4: Outlier detection
            df, outlier_stats = self.detect_outliers(df, z_threshold=3.0)
            self.validation_report["checks"]["outliers"] = outlier_stats
            
            # Check 5: Location ID validation
            df, loc_stats = self.check_location_ids(df)
            self.validation_report["checks"]["location_ids"] = loc_stats
            
            # Compute final statistics
            final_stats = self.compute_statistics(df)
            self.validation_report["statistics"] = final_stats
            
            # Summary
            self.validation_report["summary"] = {
                "records_before": original_count,
                "records_after": df.count(),
                "records_removed": original_count - df.count(),
                "retention_rate": round((df.count() / original_count) * 100, 2) if original_count > 0 else 100.0
            }
            
            # Save cleaned data
            logger.info(f"💾 Saving cleaned data to {output_path}...")
            df.write.mode("overwrite") \
                .option("compression", "snappy") \
                .parquet(output_path)
            logger.info(f"✅ Cleaned data saved")
            
            # Save validation report
            report_path = "logs/02_validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.validation_report, f, indent=2, default=str)
            logger.info(f"✅ Validation report saved to {report_path}")
            
            # Display summary
            logger.info("=" * 70)
            logger.info("📋 VALIDATION SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Records before: {self.validation_report['summary']['records_before']:,}")
            logger.info(f"Records after:  {self.validation_report['summary']['records_after']:,}")
            logger.info(f"Records removed: {self.validation_report['summary']['records_removed']:,}")
            logger.info(f"Retention rate: {self.validation_report['summary']['retention_rate']:.1f}%")
            logger.info(f"\n✨ DATA VALIDATION COMPLETE - Ready for feature engineering")
            logger.info("=" * 70)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Validation pipeline failed: {str(e)}", exc_info=True)
            self.validation_report["errors"].append(str(e))
            return 1


def main():
    """Main entry point"""
    spark = SparkSession.builder \
        .appName("Taxi-Data-Validation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", 200) \
        .config("hadoop.fs.viewfs.impl.disable.cache", "true") \
        .config("fs.viewfs.enable.inner.cache", "false") \
        .getOrCreate()
    
    validator = DataValidator(spark)
    
    input_path = "data/processed/ingested_data.parquet"
    output_path = "data/processed/validated_data.parquet"
    
    result = validator.validate_and_clean(input_path, output_path)
    
    spark.stop()
    return result


if __name__ == "__main__":
    sys.exit(main())
