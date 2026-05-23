"""
Week 1.2: Data Validation & Quality Checks (PyArrow-based, memory efficient)
Purpose: Validate ingested data, detect outliers, produce cleaned dataset
Note: Uses PyArrow compute functions - no pandas concat, handles 81M records efficiently

Input: data/processed/ingested_data.parquet (81.6M records)
Output: 
  - data/processed/validated_data.parquet (cleaned)
  - logs/02_validation_report.json (stats)

Execution: python scripts/02_data_validation_pyarrow.py
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Tuple

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np

# ============================================================================
# LOGGING SETUP
# ============================================================================

os.makedirs("logs", exist_ok=True)

class UTF8StreamHandler(logging.StreamHandler):
    """UTF8-compatible logging handler"""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg.encode('utf-8', errors='replace').decode('utf-8'))
            stream.write(self.terminator)
            stream.flush()
        except:
            self.handleError(record)

logger = logging.getLogger("DataValidation")
logger.setLevel(logging.INFO)

# Console handler
console_handler = UTF8StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler("logs/02_validation.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)
logger.addHandler(file_handler)

# ============================================================================
# DATA VALIDATOR
# ============================================================================

class DataValidator:
    def __init__(self, batch_size: int = 100000):
        self.batch_size = batch_size  # Process 100k rows at a time
        self.validation_report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "02_data_validation_pyarrow",
            "checks": {},
            "statistics": {},
            "summary": {}
        }

    def load_data_batched(self, path: str):
        """Load parquet file in batches to avoid memory overflow"""
        logger.info(f"[DIR] Loading ingested data from {path} (batch_size={self.batch_size:,})...")
        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        logger.info(f"[OK] Total {total_rows:,} records, {len(parquet_file.schema_arrow.names)} columns")
        return parquet_file, total_rows

    def check_schema(self, table: pa.Table) -> Tuple[bool, Dict]:
        """[DEPRECATED - Use check_schema_pandas instead]"""
        pass

    def check_nulls(self, table: pa.Table) -> Tuple[pa.Table, Dict]:
        """[DEPRECATED - Integrated in _validate_batch]"""
        pass

    def check_date_range(self, table: pa.Table) -> Tuple[pa.Table, Dict]:
        """[DEPRECATED - Integrated in _validate_batch]"""
        pass

    def detect_outliers(self, table: pa.Table, z_threshold: float = 3.0) -> Tuple[pa.Table, Dict]:
        """[DEPRECATED - Integrated in _validate_batch]"""
        pass

    def check_location_ids(self, table: pa.Table, valid_range: Tuple[int, int] = (1, 263)) -> Tuple[pa.Table, Dict]:
        """[DEPRECATED - Integrated in _validate_batch]"""
        pass

    def compute_statistics(self, table: pa.Table) -> Dict:
        """[DEPRECATED - Use compute_statistics_batched instead]"""
        pass

    def compute_statistics_batched(self, parquet_path: str) -> Dict:
        """Compute statistics from output file in batches (memory efficient)"""
        logger.info("📊 Computing final statistics (batched)...")
        
        # Initialize accumulators
        total_records = 0
        fare_sum = 0.0
        fare_sum_sq = 0.0  # For stddev
        fare_min = float('inf')
        fare_max = float('-inf')
        fare_count = 0
        
        distance_sum = 0.0
        distance_sum_sq = 0.0
        distance_min = float('inf')
        distance_max = float('-inf')
        distance_count = 0
        
        passenger_sum = 0.0
        passenger_min = float('inf')
        passenger_max = float('-inf')
        passenger_count = 0
        
        # Read in batches
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=100000):
                df = batch.to_pandas()
                n = len(df)
                total_records += n
                
                if n > 0 and 'fare_amount' in df.columns:
                    fare_vals = df['fare_amount'].dropna().values
                    if len(fare_vals) > 0:
                        fare_sum += fare_vals.sum()
                        fare_sum_sq += (fare_vals ** 2).sum()
                        fare_min = min(fare_min, fare_vals.min())
                        fare_max = max(fare_max, fare_vals.max())
                        fare_count += len(fare_vals)
                
                if n > 0 and 'trip_distance' in df.columns:
                    dist_vals = df['trip_distance'].dropna().values
                    if len(dist_vals) > 0:
                        distance_sum += dist_vals.sum()
                        distance_sum_sq += (dist_vals ** 2).sum()
                        distance_min = min(distance_min, dist_vals.min())
                        distance_max = max(distance_max, dist_vals.max())
                        distance_count += len(dist_vals)
                
                if n > 0 and 'passenger_count' in df.columns:
                    pass_vals = df['passenger_count'].dropna().values
                    if len(pass_vals) > 0:
                        passenger_sum += pass_vals.sum()
                        passenger_min = min(passenger_min, pass_vals.min())
                        passenger_max = max(passenger_max, pass_vals.max())
                        passenger_count += len(pass_vals)
            
            # Calculate means and stddevs using proper counts
            fare_mean = fare_sum / fare_count if fare_count > 0 else 0.0
            fare_variance = (fare_sum_sq / fare_count - fare_mean ** 2) if fare_count > 0 else 0.0
            fare_stddev = np.sqrt(max(fare_variance, 0.0))
            
            distance_mean = distance_sum / distance_count if distance_count > 0 else 0.0
            distance_variance = (distance_sum_sq / distance_count - distance_mean ** 2) if distance_count > 0 else 0.0
            distance_stddev = np.sqrt(max(distance_variance, 0.0))
            
            passenger_mean = passenger_sum / passenger_count if passenger_count > 0 else 0.0
            
            stats = {
                "record_count": total_records,
                "fare_stats": {
                    "mean": round(fare_mean, 2),
                    "stddev": round(fare_stddev, 2),
                    "min": round(fare_min, 2) if fare_min != float('inf') else 0.0,
                    "max": round(fare_max, 2) if fare_max != float('-inf') else 0.0
                },
                "distance_stats": {
                    "mean": round(distance_mean, 2),
                    "stddev": round(distance_stddev, 2),
                    "min": round(distance_min, 2) if distance_min != float('inf') else 0.0,
                    "max": round(distance_max, 2) if distance_max != float('-inf') else 0.0
                },
                "passenger_stats": {
                    "mean": round(passenger_mean, 2),
                    "min": int(passenger_min) if passenger_min != float('inf') else 0,
                    "max": int(passenger_max) if passenger_max != float('-inf') else 0
                }
            }
            
            logger.info(f"  - Fare: μ={stats['fare_stats']['mean']}, σ={stats['fare_stats']['stddev']}")
            logger.info(f"  - Distance: μ={stats['distance_stats']['mean']}, σ={stats['distance_stats']['stddev']}")
            logger.info(f"  - Passengers: μ={stats['passenger_stats']['mean']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"⚠️  Failed to compute statistics: {str(e)}")
            return {"record_count": total_records}

    def validate_and_clean(self, input_path: str, output_path: str) -> int:
        """Execute complete validation pipeline with batch processing"""
        logger.info("=" * 70)
        logger.info("🚕 NYC TAXI DATA VALIDATION PIPELINE (Week 1.2 - Step 2)")
        logger.info("=" * 70)
        
        try:
            # Load data metadata
            parquet_file, total_rows = self.load_data_batched(input_path)
            original_count = total_rows
            
            # Initialize batch processor
            batch_writer = None
            all_stats = {
                "schema": None,
                "nulls": {"before_count": 0, "after_count": 0, "dropped_rows": 0, "null_counts": {}},
                "date_range": {"before_count": 0, "after_count": 0, "out_of_range_count": 0},
                "outliers": {"before_count": 0, "after_count": 0, "total_outliers": 0, "outliers_by_column": {}},
                "location_ids": {"before_count": 0, "after_count": 0, "invalid_pu_count": 0, "invalid_do_count": 0}
            }
            
            processed_rows = 0
            batch_num = 0
            
            # Process file in batches
            logger.info(f"🔄 Processing {total_rows:,} records in batches of {self.batch_size:,}...")
            
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                batch_num += 1
                table = batch.to_pandas().reset_index(drop=True)
                
                if len(table) == 0:
                    continue
                
                batch_start = processed_rows
                
                # Check 1: Schema validation (only first batch)
                if batch_num == 1:
                    schema_ok, schema_check = self.check_schema_pandas(table)
                    all_stats["schema"] = schema_check
                    if not schema_ok:
                        logger.error("❌ Schema validation failed!")
                        return 1
                
                # Check 2-5: Apply all validations
                table = self._validate_batch(table, all_stats)
                
                # Write batch to output
                if len(table) > 0:
                    table_arrow = pa.Table.from_pandas(table)
                    if batch_writer is None:
                        batch_writer = pq.ParquetWriter(output_path, table_arrow.schema)
                    batch_writer.write_table(table_arrow)
                
                processed_rows += len(batch)
                retained = len(table)
                pct = (processed_rows / total_rows) * 100
                logger.info(f"  Batch {batch_num}: processed {processed_rows:,}/{total_rows:,} rows ({pct:.1f}%) | retained {retained:,}")
            
            # Finalize writer
            if batch_writer is not None:
                batch_writer.close()
            
            # Compute final statistics from output file (in batches)
            final_stats = self.compute_statistics_batched(output_path)
            self.validation_report["statistics"] = final_stats
            
            # Summary
            final_count = final_stats.get("record_count", 0)
            self.validation_report["checks"]["nulls"] = all_stats["nulls"]
            self.validation_report["checks"]["date_range"] = all_stats["date_range"]
            self.validation_report["checks"]["outliers"] = all_stats["outliers"]
            self.validation_report["checks"]["location_ids"] = all_stats["location_ids"]
            self.validation_report["checks"]["schema"] = all_stats["schema"]
            
            self.validation_report["summary"] = {
                "records_before": original_count,
                "records_after": final_count,
                "records_removed": original_count - final_count,
                "retention_rate": round((final_count / original_count) * 100, 2) if original_count > 0 else 100.0
            }
            
            # Save validation report
            report_path = "logs/02_validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.validation_report, f, indent=2, default=str)
            logger.info(f"✅ Validation report saved to {report_path}")
            
            # Display summary
            logger.info("=" * 70)
            logger.info("📋 VALIDATION SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Records before: {original_count:,}")
            logger.info(f"Records after:  {final_count:,}")
            logger.info(f"Records removed: {original_count - final_count:,}")
            logger.info(f"Retention rate:  {self.validation_report['summary']['retention_rate']:.1f}%")
            logger.info(f"Processed in {batch_num} batches")
            logger.info("=" * 70)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Validation pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    def check_schema_pandas(self, df) -> Tuple[bool, Dict]:
        """Validate schema using pandas"""
        logger.info("🔍 Checking schema...")
        
        required_cols = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'pulocationid', 'dolocationid',
            'fare_amount', 'passenger_count', 'trip_distance',
            'payment_type', 'total_amount'
        ]
        
        schema_check = {
            "total_columns": len(df.columns),
            "expected_columns": required_cols,
            "missing_columns": [],
            "valid": True,
            "actual_columns": list(df.columns)
        }
        
        for col in required_cols:
            if col not in df.columns:
                schema_check["missing_columns"].append(col)
                schema_check["valid"] = False
                logger.warning(f"  ⚠️  Missing column: {col}")
        
        if schema_check["valid"]:
            logger.info(f"✅ Schema valid: all {len(required_cols)} required columns present")
        
        return schema_check["valid"], schema_check
    
    def _validate_batch(self, df, stats_accumulator) -> pa.Table:
        """Apply all validations to a batch"""
        before_count = len(df)
        
        # Check nulls in required fields
        required_fields = ['tpep_pickup_datetime', 'pulocationid', 'dolocationid', 'fare_amount']
        df = df.dropna(subset=required_fields)
        
        # Update null stats
        stats_accumulator["nulls"]["before_count"] += before_count
        stats_accumulator["nulls"]["after_count"] += len(df)
        stats_accumulator["nulls"]["dropped_rows"] += (before_count - len(df))
        
        before_null = len(df)
        
        # Date range filter
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df = df[(df['tpep_pickup_datetime'] >= '2019-01-01') & 
                (df['tpep_pickup_datetime'] <= '2020-06-30')]
        
        stats_accumulator["date_range"]["before_count"] += before_null
        stats_accumulator["date_range"]["after_count"] += len(df)
        stats_accumulator["date_range"]["out_of_range_count"] += (before_null - len(df))
        
        before_outlier = len(df)
        
        # Outlier detection (Z-score)
        for col in ['fare_amount', 'trip_distance', 'passenger_count']:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    if std_val > 0:
                        z_scores = np.abs((df[col] - mean_val) / std_val)
                        df = df[z_scores <= 3.0]
                    removed = before_outlier - len(df)
                    if removed > 0:
                        stats_accumulator["outliers"]["outliers_by_column"][col] = \
                            stats_accumulator["outliers"]["outliers_by_column"].get(col, 0) + removed
                    before_outlier = len(df)
        
        stats_accumulator["outliers"]["before_count"] += before_outlier + (before_outlier - len(df))
        stats_accumulator["outliers"]["after_count"] += len(df)
        stats_accumulator["outliers"]["total_outliers"] += (before_outlier - len(df))
        
        # Location ID validation
        before_loc = len(df)
        df = df[(df['pulocationid'] >= 1) & (df['pulocationid'] <= 263) &
                (df['dolocationid'] >= 1) & (df['dolocationid'] <= 263)]
        stats_accumulator["location_ids"]["before_count"] += before_loc
        stats_accumulator["location_ids"]["after_count"] += len(df)
        stats_accumulator["location_ids"]["invalid_pu_count"] += (before_loc - len(df))
        
        return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    validator = DataValidator()
    exit_code = validator.validate_and_clean(
        input_path="data/processed/ingested_data.parquet",
        output_path="data/processed/validated_data.parquet"
    )
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
