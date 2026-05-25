"""
Week 2.1: Baseline Method - Zone-Based Aggregation
Purpose: Extract baseline features by aggregating taxi trips into 15-minute windows per zone
Input: data/processed/validated_data.parquet (73.6M cleaned records)
Output: data/processed/baseline_features/ (zone-time demand features)

Execution: python scripts/03_baseline_features.py
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# ============================================================================
# LOGGING SETUP
# ============================================================================

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("BaselineFeatures")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler("logs/03_baseline_features.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)
logger.addHandler(file_handler)

# ============================================================================
# BASELINE FEATURE EXTRACTOR
# ============================================================================

class BaselineFeatureExtractor:
    def __init__(self):
        self.feature_report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "03_baseline_features",
            "config": {
                "engine": "Pandas + PyArrow",
                "batch_size": 100000,
                "temporal_granularity": "15 minutes",
                "compression": "snappy"
            },
            "statistics": {},
            "summary": {}
        }

    def load_data(self, input_path: str):
        """Load validated data in batches to avoid memory overflow"""
        logger.info(f"[DIR] Loading validated data from {input_path}...")
        
        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows
        logger.info(f"[OK] Total {total_rows:,} records, reading in batches...")
        
        return parquet_file, total_rows

    def extract_baseline_features_batched(self, parquet_file, batch_size=100000):
        """
        Extract baseline features in batches:
        - Aggregate by (PULocationID, 15-min time window)
        - Compute: demand_count, avg_fare, avg_distance, avg_passenger, etc.
        """
        logger.info(f"🔄 Extracting baseline features (batch_size={batch_size:,})...")
        
        all_features = []
        batch_num = 0
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_num += 1
            df = batch.to_pandas()
            
            if len(df) == 0:
                continue
            
            # Convert datetime
            df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            
            # Create time bucket (15-minute window)
            df['time_bucket'] = df['tpep_pickup_datetime'].dt.floor('15min')
            
            # Extract temporal features
            df['date_day'] = (df['tpep_pickup_datetime'].dt.dayofweek).astype(int)  # 0=Monday
            df['hour'] = df['tpep_pickup_datetime'].dt.hour.astype(int)
            df['date_str'] = df['tpep_pickup_datetime'].dt.date
            
            # Group by zone and time bucket and other temporal features
            batch_features = df.groupby(
                ['date_str', 'time_bucket', 'pulocationid', 'date_day', 'hour'],
                observed=True
            ).agg(
                demand_count=('tpep_pickup_datetime', 'count'),
                avg_fare=('fare_amount', 'mean'),
                sum_fare=('fare_amount', 'sum'),
                min_fare=('fare_amount', 'min'),
                max_fare=('fare_amount', 'max'),
                stddev_fare=('fare_amount', 'std'),
                avg_distance=('trip_distance', 'mean'),
                median_distance=('trip_distance', lambda x: x.quantile(0.5)),
                p95_distance=('trip_distance', lambda x: x.quantile(0.95)),
                avg_passenger=('passenger_count', 'mean')
            ).reset_index()
            
            # Rename for consistency
            batch_features = batch_features.rename(columns={'pulocationid': 'zone_id'})
            
            # Add window start/end
            batch_features['window_start'] = batch_features['time_bucket']
            batch_features['window_end'] = batch_features['time_bucket'] + pd.Timedelta(minutes=15)
            
            all_features.append(batch_features)
            
            if batch_num % 5 == 0:
                logger.info(f"  Batch {batch_num} processed, accumulated {sum(len(f) for f in all_features):,} feature records")
        
        # Concatenate all batches
        if all_features:
            features_df = pd.concat(all_features, ignore_index=True)
            logger.info(f"✅ Baseline features extracted: {len(features_df):,} records")
            return features_df
        else:
            logger.error("❌ No features extracted!")
            return pd.DataFrame()

    def compute_statistics(self, features_df):
        """Compute statistics on baseline features"""
        logger.info("📊 Computing feature statistics...")
        
        stats = {
            "total_records": len(features_df),
            "unique_zones": features_df['zone_id'].nunique(),
            "time_range": {},
            "demand_stats": {},
            "fare_stats": {},
            "distance_stats": {},
            "passenger_stats": {}
        }
        
        # Time range
        stats["time_range"] = {
            "start": str(features_df['window_start'].min()),
            "end": str(features_df['window_end'].max())
        }
        
        # Demand statistics
        stats["demand_stats"] = {
            "mean": round(features_df['demand_count'].mean(), 2),
            "median": int(features_df['demand_count'].median()),
            "min": int(features_df['demand_count'].min()),
            "max": int(features_df['demand_count'].max()),
            "p05": int(features_df['demand_count'].quantile(0.05)),
            "p95": int(features_df['demand_count'].quantile(0.95))
        }
        
        # Fare statistics
        stats["fare_stats"] = {
            "mean": round(features_df['avg_fare'].mean(), 2),
            "median": round(features_df['avg_fare'].median(), 2),
            "min": round(features_df['min_fare'].min(), 2),
            "max": round(features_df['max_fare'].max(), 2)
        }
        
        # Distance statistics
        stats["distance_stats"] = {
            "mean": round(features_df['avg_distance'].mean(), 2),
            "median": round(features_df['median_distance'].median(), 2),
            "min": round(features_df['median_distance'].min(), 2),
            "max": round(features_df['p95_distance'].max(), 2)
        }
        
        # Passenger statistics
        stats["passenger_stats"] = {
            "mean": round(features_df['avg_passenger'].mean(), 2)
        }
        
        logger.info(f"  - Total records: {stats['total_records']:,}")
        logger.info(f"  - Unique zones: {stats['unique_zones']}")
        logger.info(f"  - Time range: {stats['time_range']['start']} to {stats['time_range']['end']}")
        logger.info(f"  - Demand: μ={stats['demand_stats']['mean']}, median={stats['demand_stats']['median']}")
        logger.info(f"  - Fare: μ={stats['fare_stats']['mean']}, min={stats['fare_stats']['min']}, max={stats['fare_stats']['max']}")
        logger.info(f"  - Distance: μ={stats['distance_stats']['mean']}, median={stats['distance_stats']['median']}")
        logger.info(f"  - Passenger: μ={stats['passenger_stats']['mean']}")
        
        return stats

    def save_features(self, features_df, output_path: str):
        """Save baseline features to Parquet"""
        logger.info(f"💾 Saving baseline features to {output_path}...")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Write using pandas to_parquet
        features_df.to_parquet(
            os.path.join(output_path, 'baseline_features.parquet'),
            compression='snappy',
            index=False
        )
        
        logger.info(f"✅ Baseline features saved")

    def run(self, input_path: str, output_path: str) -> int:
        """Execute baseline feature extraction pipeline"""
        logger.info("=" * 70)
        logger.info("🚕 NYC TAXI BASELINE FEATURES EXTRACTION (Week 2.1 - Pandas)")
        logger.info("=" * 70)
        
        try:
            # Load validated data
            parquet_file, record_count = self.load_data(input_path)
            
            # Extract baseline features in batches
            baseline_features = self.extract_baseline_features_batched(parquet_file)
            
            if len(baseline_features) == 0:
                logger.error("❌ No features extracted!")
                return 1
            
            # Compute statistics
            stats = self.compute_statistics(baseline_features)
            self.feature_report["statistics"] = stats
            
            # Summary
            self.feature_report["summary"] = {
                "input_records": record_count,
                "output_records": stats["total_records"],
                "unique_zones_covered": stats["unique_zones"],
                "feature_count": len(baseline_features.columns),
                "features": list(baseline_features.columns)
            }
            
            # Save features
            self.save_features(baseline_features, output_path)
            
            # Save feature report
            report_path = "logs/03_baseline_features_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.feature_report, f, indent=2, default=str)
            logger.info(f"✅ Feature report saved to {report_path}")
            
            # Display summary
            logger.info("=" * 70)
            logger.info("📋 BASELINE FEATURES SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Input records: {record_count:,}")
            logger.info(f"Output feature records: {stats['total_records']:,}")
            logger.info(f"Zones covered: {stats['unique_zones']}/263")
            logger.info(f"Time range: {stats['time_range']['start']} → {stats['time_range']['end']}")
            logger.info(f"Features extracted: {len(baseline_features.columns)}")
            logger.info(f"  {', '.join(baseline_features.columns)}")
            logger.info("=" * 70)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Baseline feature extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1

# ============================================================================
# MAIN
# ============================================================================

def main():
    extractor = BaselineFeatureExtractor()
    exit_code = extractor.run(
        input_path="data/processed/validated_data.parquet",
        output_path="data/processed/baseline_features"
    )
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
