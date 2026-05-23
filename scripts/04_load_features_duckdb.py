#!/usr/bin/env python
"""
Alternative feature loader using DuckDB (local SQL database)
This bypasses Cassandra when it's unavailable
"""
import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import pyarrow.parquet as pq
import duckdb

# Setup logging - No emojis for Windows console compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/04_load_features_duckdb.log')
    ],
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

class DuckDBFeatureLoader:
    """Load baseline features to DuckDB (local SQL DB alternative to Cassandra)"""
    
    def __init__(self, db_path='data/processed/taxi_features.duckdb'):
        self.db_path = db_path
        self.conn = None
        self.load_report = {
            "timestamp": datetime.now().isoformat(),
            "database": "DuckDB",
            "reason": "Cassandra unavailable - using local SQL alternative"
        }
    
    def connect(self):
        """Connect to DuckDB"""
        logger.info(f"[DuckDB] Opening database at {self.db_path}...")
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info("[DuckDB] Connected successfully")
            self._create_schema()
            return True
        except Exception as e:
            logger.error(f"[DuckDB] Connection failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_schema(self):
        """Create baseline_features table"""
        logger.info("[DuckDB] Creating baseline_features table...")
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS baseline_features (
                    date_str DATE,
                    time_bucket TIMESTAMP,
                    zone_id INTEGER,
                    date_day INTEGER,
                    hour INTEGER,
                    demand_count BIGINT,
                    avg_fare FLOAT,
                    sum_fare FLOAT,
                    min_fare FLOAT,
                    max_fare FLOAT,
                    stddev_fare FLOAT,
                    avg_distance FLOAT,
                    median_distance FLOAT,
                    p95_distance FLOAT,
                    avg_passenger FLOAT,
                    window_start TIMESTAMP,
                    window_end TIMESTAMP
                )
            """)
            # Create indexes for common queries
            try:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_zone_window ON baseline_features(zone_id, window_start)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_date_hour ON baseline_features(date_str, hour)")
                logger.info("[DuckDB] Schema created/verified with indexes")
            except:
                logger.info("[DuckDB] Schema created/verified")
        except Exception as e:
            logger.warning(f"[DuckDB] Schema creation warning: {str(e)}")
    
    def load_parquet_batched(self, input_path: str, batch_size: int = 100000):
        """Load Parquet file in batches"""
        logger.info(f"[Parquet] Loading from {input_path}...")
        
        try:
            parquet_file = pq.ParquetFile(input_path)
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"[Parquet] Total rows: {total_rows:,}")
            
            all_batches = []
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
                df = batch.to_pandas()
                all_batches.append(df)
                if i % 10 == 0:
                    logger.info(f"[Parquet] Loaded {i+1} batches...")
            
            if all_batches:
                df_combined = pd.concat(all_batches, ignore_index=True)
                logger.info(f"[Parquet] Loaded {len(df_combined):,} rows total")
                return df_combined
            else:
                logger.error("[Parquet] No data loaded")
                return None
        except Exception as e:
            logger.error(f"[Parquet] Error loading: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def insert_features_bulk(self, df):
        """Insert features using DuckDB bulk insert"""
        logger.info(f"[Insert] Inserting {len(df):,} features into DuckDB...")
        
        try:
            # DuckDB bulk insert is very fast
            logger.info("[Insert] Registering DataFrame...")
            self.conn.register('temp_df', df)
            
            logger.info("[Insert] Executing INSERT statement...")
            self.conn.execute("""
                INSERT INTO baseline_features 
                SELECT * FROM temp_df
            """)
            logger.info(f"[Insert] Inserted {len(df):,} rows successfully")
            
            # Get actual count
            result = self.conn.execute("SELECT COUNT(*) FROM baseline_features").fetchall()
            total_count = result[0][0]
            logger.info(f"[Insert] Total in database: {total_count:,}")
            
            return len(df)
        except Exception as e:
            logger.error(f"[Insert] Insert failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0
    
    def verify_load(self):
        """Verify data was loaded"""
        logger.info("[Verify] Verifying data load...")
        
        try:
            # Count total rows
            result = self.conn.execute("SELECT COUNT(*) FROM baseline_features").fetchall()
            total_count = result[0][0]
            logger.info(f"[Verify] Total rows: {total_count:,}")
            
            # Check zones
            result = self.conn.execute("SELECT COUNT(DISTINCT zone_id) FROM baseline_features").fetchall()
            zones = result[0][0]
            logger.info(f"[Verify] Unique zones: {zones}")
            
            # Check date range
            result = self.conn.execute("""
                SELECT MIN(window_start), MAX(window_start) FROM baseline_features
            """).fetchall()
            min_time = result[0][0]
            max_time = result[0][1]
            logger.info(f"[Verify] Time range: {min_time} to {max_time}")
            
            # Sample query
            result = self.conn.execute("SELECT * FROM baseline_features WHERE zone_id = 1 LIMIT 5").fetchall()
            logger.info(f"[Verify] Sample query returned {len(result)} rows")
            
            self.load_report["statistics"] = {
                "total_rows": total_count,
                "unique_zones": zones,
                "time_range_start": str(min_time),
                "time_range_end": str(max_time),
                "sample_rows": len(result)
            }
            
            logger.info("[Verify] Verification complete")
            return True
        except Exception as e:
            logger.error(f"[Verify] Verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def disconnect(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
        logger.info("[DuckDB] Disconnected")
    
    def load(self, input_path: str):
        """Execute full load"""
        logger.info("=" * 70)
        logger.info("LOAD BASELINE FEATURES TO DUCKDB (Local SQL Alternative)")
        logger.info("=" * 70)
        
        try:
            # Connect
            if not self.connect():
                return 1
            
            # Load Parquet
            df = self.load_parquet_batched(input_path)
            if df is None:
                return 1
            
            # Insert into DuckDB
            rows_inserted = self.insert_features_bulk(df)
            
            # Verify
            if not self.verify_load():
                logger.warning("[Verify] Verification had issues")
            
            # Summary
            self.load_report["summary"] = {
                "input_file": input_path,
                "rows_loaded": rows_inserted,
                "database_file": self.db_path
            }
            
            # Save report
            report_path = "logs/04_load_features_duckdb_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.load_report, f, indent=2, default=str)
            logger.info(f"[Report] Load report saved to {report_path}")
            
            logger.info("=" * 70)
            logger.info("LOAD COMPLETE - Features in DuckDB!")
            logger.info(f"Database location: {self.db_path}")
            logger.info("=" * 70)
            
            return 0
        
        except Exception as e:
            logger.error(f"[Error] Load failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
        
        finally:
            self.disconnect()

def main():
    loader = DuckDBFeatureLoader(
        db_path='data/processed/taxi_features.duckdb'
    )
    
    exit_code = loader.load("data/processed/baseline_features/baseline_features.parquet")
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
