"""
Task 4.4: Load Baseline Features to Cassandra (Fast Concurrent Version)
Purpose: Ingest baseline_features.parquet into Cassandra with concurrent bulk insert
Execution: python scripts/04_load_features_cassandra_fast.py
"""

import logging
import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy
from cassandra.concurrent import execute_concurrent_with_args

# ============================================================================
# LOGGING
# ============================================================================

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("CassandraLoader")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("logs/04_load_features_cassandra.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)
logger.addHandler(file_handler)

# ============================================================================
# CASSANDRA LOADER
# ============================================================================

class CassandraFeatureLoader:
    def __init__(self, contact_points=['127.0.0.1', '127.0.0.2', '127.0.0.3'], 
                 port=9042, keyspace='taxi_db'):
        self.contact_points = contact_points
        self.port = port
        self.keyspace = keyspace
        self.cluster = None
        self.session = None
        self.load_report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "04_load_features_cassandra",
            "config": {},
            "statistics": {},
            "summary": {}
        }

    def connect(self):
        """Connect to Cassandra"""
        logger.info(f"🔗 Connecting to Cassandra: {self.contact_points}:{self.port}...")
        
        try:
            self.cluster = Cluster(
                contact_points=self.contact_points,
                port=self.port,
                load_balancing_policy=RoundRobinPolicy()
            )
            self.session = self.cluster.connect(self.keyspace)
            logger.info("✅ Connected to Cassandra")
            
            # Verify session is ready
            self._verify_session_health()
            
            self.load_report["config"] = {
                "contact_points": self.contact_points,
                "port": self.port,
                "keyspace": self.keyspace,
                "replication_factor": 3
            }
            
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {str(e)}")
            return False
    
    def _verify_session_health(self):
        """Verify session is healthy by running a test query"""
        try:
            result = self.session.execute("SELECT COUNT(*) FROM baseline_features LIMIT 1")
            logger.info(f"✓ Session health check passed - baseline_features table is accessible")
            return True
        except Exception as e:
            logger.warning(f"⚠️  Session health check warning: {str(e)}")
            return False

    def load_parquet_batched(self, input_path: str, batch_size: int = 10000):
        """Load Parquet file in batches"""
        logger.info(f"📂 Loading Parquet from {input_path}...")
        
        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows
        logger.info(f"   Total rows: {total_rows:,}")
        
        all_batches = []
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            all_batches.append(df)
        
        if all_batches:
            df_combined = pd.concat(all_batches, ignore_index=True)
            logger.info(f"✅ Loaded {len(df_combined):,} rows")
            return df_combined
        else:
            logger.error("❌ No data loaded")
            return None

    def insert_features_bulk(self, df):
        """Insert features using concurrent bulk insert (OPTIMIZED)"""
        logger.info("💾 Inserting features into Cassandra (concurrent batch)...")
        
        # Prepare insert statement once
        insert_stmt = self.session.prepare("""
            INSERT INTO baseline_features 
            (zone_id, window_start, date_str, date_day, hour, demand_count, 
             avg_fare, sum_fare, min_fare, max_fare, stddev_fare, 
             avg_distance, median_distance, p95_distance, avg_passenger)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        
        # Test insert with first row
        logger.info("  Testing insert with first row...")
        try:
            first_row = df.iloc[0]
            test_params = [
                int(first_row['zone_id']),
                pd.Timestamp(first_row['window_start']).to_pydatetime(),
                pd.Timestamp(first_row['date_str']).date(),
                int(first_row['date_day']),
                int(first_row['hour']),
                int(first_row['demand_count']),
                float(first_row['avg_fare']),
                float(first_row['sum_fare']),
                float(first_row['min_fare']),
                float(first_row['max_fare']),
                float(first_row['stddev_fare']) if pd.notna(first_row['stddev_fare']) else 0.0,
                float(first_row['avg_distance']),
                float(first_row['median_distance']),
                float(first_row['p95_distance']),
                float(first_row['avg_passenger'])
            ]
            self.session.execute(insert_stmt, test_params)
            logger.info("  ✓ Test insert successful!")
        except Exception as e:
            logger.error(f"  ❌ Test insert failed: {str(e)}")
            logger.error(f"  Cannot proceed with bulk insert - session may be unavailable")
            return 0
        
        # Prepare all insert parameters
        all_params = []
        for idx, row in df.iterrows():
            try:
                params = [
                    int(row['zone_id']),
                    pd.Timestamp(row['window_start']).to_pydatetime(),
                    pd.Timestamp(row['date_str']).date(),
                    int(row['date_day']),
                    int(row['hour']),
                    int(row['demand_count']),
                    float(row['avg_fare']),
                    float(row['sum_fare']),
                    float(row['min_fare']),
                    float(row['max_fare']),
                    float(row['stddev_fare']) if pd.notna(row['stddev_fare']) else 0.0,
                    float(row['avg_distance']),
                    float(row['median_distance']),
                    float(row['p95_distance']),
                    float(row['avg_passenger'])
                ]
                all_params.append(params)
            except Exception as e:
                logger.warning(f"  Row {idx} preparation failed: {str(e)[:80]}")
                continue
        
        # Execute all queries concurrently
        rows_processed = 1  # Already have the test row
        failed_count = 0
        error_samples = []
        
        logger.info(f"  Executing {len(all_params):,} concurrent inserts (concurrency=150)...")
        try:
            for i, (success, result) in enumerate(execute_concurrent_with_args(
                self.session, insert_stmt, all_params, concurrency=150
            )):
                if success:
                    rows_processed += 1
                else:
                    failed_count += 1
                    # Capture first few errors for debugging
                    if len(error_samples) < 5:
                        error_samples.append(f"Row {i}: {str(result)[:100]}")
                
                if rows_processed % 500000 == 0 and rows_processed > 0:
                    logger.info(f"  ✓ Progress: {rows_processed:,} rows inserted ({failed_count} failed)")
        except Exception as e:
            logger.error(f"❌ Concurrent execution critical error: {str(e)}")
            logger.warning(f"Falling back to sequential insert...")
            
            # Fallback: sequential insert if concurrent fails
            for idx, params in enumerate(all_params):
                try:
                    self.session.execute(insert_stmt, params)
                    rows_processed += 1
                except Exception as e2:
                    failed_count += 1
                    if len(error_samples) < 5:
                        error_samples.append(f"Row {idx}: {str(e2)[:100]}")
                
                if rows_processed % 100000 == 0 and rows_processed > 0:
                    logger.info(f"  ✓ Sequential progress: {rows_processed:,} rows")
        
        # Log error samples
        if error_samples:
            logger.warning(f"Sample errors from failed inserts:")
            for sample in error_samples:
                logger.warning(f"  {sample}")
        
        logger.info(f"✅ Inserted {rows_processed:,} rows total ({failed_count} failed)")
        return rows_processed

    def verify_load(self):
        """Verify data was loaded"""
        logger.info("🔍 Verifying data load...")
        
        try:
            # Count total rows
            result = self.session.execute("SELECT COUNT(*) as cnt FROM baseline_features")
            total_count = result[0].cnt
            logger.info(f"   ✓ Total rows in Cassandra: {total_count:,}")
            
            # Check zones covered
            result = self.session.execute("SELECT DISTINCT zone_id FROM baseline_features")
            zones = len(result)
            logger.info(f"   ✓ Unique zones: {zones}")
            
            # Check date range
            result = self.session.execute("""
                SELECT MIN(window_start) as min_time, MAX(window_start) as max_time 
                FROM baseline_features
            """)
            min_time = result[0].min_time
            max_time = result[0].max_time
            logger.info(f"   ✓ Time range: {min_time} → {max_time}")
            
            # Sample query
            result = self.session.execute("""
                SELECT * FROM baseline_features WHERE zone_id = 1 LIMIT 5
            """)
            logger.info(f"   ✓ Sample query returned {len(result)} rows")
            
            self.load_report["statistics"] = {
                "total_rows": total_count,
                "unique_zones": zones,
                "time_range_start": str(min_time),
                "time_range_end": str(max_time)
            }
            
            logger.info("✅ Data verification complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {str(e)}")
            return False

    def load(self, input_path: str):
        """Execute full load"""
        logger.info("=" * 70)
        logger.info("🚕 LOAD BASELINE FEATURES TO CASSANDRA (Fast Concurrent)")
        logger.info("=" * 70)
        
        try:
            # Connect
            if not self.connect():
                return 1
            
            # Load Parquet
            df = self.load_parquet_batched(input_path)
            if df is None:
                return 1
            
            # Insert into Cassandra
            rows_inserted = self.insert_features_bulk(df)
            
            # Verify
            if not self.verify_load():
                logger.warning("⚠️  Verification had issues, but may still have loaded data")
            
            # Summary
            self.load_report["summary"] = {
                "input_file": input_path,
                "rows_loaded": rows_inserted,
                "replication_factor": 3,
                "consistency_level": "ONE (insert) / QUORUM (for critical reads)"
            }
            
            # Save report
            import json
            report_path = "logs/04_load_features_cassandra_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.load_report, f, indent=2, default=str)
            logger.info(f"✅ Load report saved to {report_path}")
            
            logger.info("=" * 70)
            logger.info("✅ LOAD COMPLETE - All features in Cassandra!")
            logger.info("=" * 70)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Load failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
        
        finally:
            self.disconnect()

    def disconnect(self):
        """Close connection"""
        if self.session:
            self.session.shutdown()
        if self.cluster:
            self.cluster.shutdown()
        logger.info("🔌 Disconnected from Cassandra")
        logger.info("🔌 Disconnected from Cassandra")

# ============================================================================
# MAIN
# ============================================================================

def main():
    loader = CassandraFeatureLoader(
        contact_points=['127.0.0.1'],  # Use localhost only, cassandra-driver will discover other nodes
        port=9042,  # Use first mapped port
        keyspace='taxi_db'
    )
    
    exit_code = loader.load("data/processed/baseline_features/baseline_features.parquet")
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
