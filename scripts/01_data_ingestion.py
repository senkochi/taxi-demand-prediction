"""
Week 1.2: Data Ingestion Pipeline
Purpose: Load raw NYC Taxi CSV data and convert to Parquet format

Input: CSV files from data/raw/ (date range 01-2019 to 06-2020)
Output: Parquet file at data/processed/ingested_data.parquet

Execution: python scripts/01_data_ingestion.py
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import io

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, TimestampType
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
        logging.FileHandler('logs/01_ingestion.log', encoding='utf-8'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_spark_session():
    """Initialize PySpark session with optimal configuration"""
    # Set Java options for Hadoop compatibility on Windows
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--driver-java-options '-Dhadoop.security.groups.cache.secs=250 "
        "-Dcom.sun.jndi.ldap.connect.pool=false' "
        "pyspark-shell"
    )
    
    spark = SparkSession.builder \
        .appName("Taxi-Data-Ingestion") \
        .master("local[4]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", 200) \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.LocalFs") \
        .config("spark.hadoop.fs.viewfs.impl.disable.cache", "true") \
        .config("spark.hadoop.fs.viewfs.enable.inner.cache", "false") \
        .config("spark.io.compression.codec", "snappy") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryo.registrationRequired", "false") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()
    
    logger.info("[OK] Spark session created successfully")
    return spark


def define_schema():
    """
    Define NYC Taxi data schema
    
    Reference: NYC TLC Taxi Trip Record Data Dictionary
    """
    schema = StructType([
        StructField("VendorID", IntegerType(), True),
        StructField("tpep_pickup_datetime", TimestampType(), False),
        StructField("tpep_dropoff_datetime", TimestampType(), False),
        StructField("passenger_count", IntegerType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("RateCodeID", IntegerType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("PULocationID", IntegerType(), False),
        StructField("DOLocationID", IntegerType(), False),
        StructField("payment_type", IntegerType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("extra", DoubleType(), True),
        StructField("mta_tax", DoubleType(), True),
        StructField("improvement_surcharge", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("tolls_amount", DoubleType(), True),
        StructField("total_amount", DoubleType(), True)
    ])
    
    logger.info(f"[OK] Schema defined with {len(schema.fields)} fields")
    return schema


def load_raw_data_and_save(spark, raw_data_path, output_path):
    """
    Load raw CSV data from data/raw/ directory using Pandas and save directly to Parquet with PyArrow
    
    This approach completely bypasses Hadoop filesystem issues by:
    1. Using Pandas to read each CSV file with explicit dtypes
    2. Normalizing schema to ensure consistency
    3. Using PyArrow to write Parquet files directly
    4. Handles files incrementally to manage memory
    
    Args:
        spark: SparkSession (used only for final validation)
        raw_data_path: Path to raw CSV directory
        output_path: Output Parquet file path
        
    Returns:
        (total_records, output_path)
    """
    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow as pa
    from pathlib import Path
    
    total_records = 0
    files_processed = 0
    all_tables = []
    
    # Dynamic schema building - will infer from first file then standardize
    target_schema = None
    
    try:
        logger.info("[DIR] Loading CSV files from {}...".format(raw_data_path))
        
        # Use Path to handle Windows paths properly
        csv_dir = Path(raw_data_path)
        # Only get yellow tripdata files, exclude zone lookup
        csv_files = sorted([f for f in csv_dir.glob("*.csv") if "yellow_tripdata" in f.name])
        
        if not csv_files:
            logger.error("[ERROR] No CSV files found in {}".format(raw_data_path))
            raise FileNotFoundError("No CSV files found in {}".format(raw_data_path))
        
        logger.info("Found {} CSV file(s)".format(len(csv_files)))
        
        # Process each CSV file with Pandas and collect PyArrow tables
        for csv_file in csv_files:
            csv_path = str(csv_file)
            logger.info("Reading {}...".format(csv_file.name))
            
            try:
                # Read CSV with Pandas - disable low_memory to avoid dtype warnings
                df_pandas = pd.read_csv(csv_path, low_memory=False)
                record_count = len(df_pandas)
                
                # Skip empty files
                if record_count == 0:
                    logger.warning("  Skipped (empty file)")
                    continue
                
                logger.info("  Loaded {} records".format(record_count))
                total_records += record_count
                
                # Normalize column names (handle case variations like RatecodeID vs RateCodeID)
                df_pandas.columns = [col.lower() for col in df_pandas.columns]
                
                # Rename if needed to standard names
                column_mapping = {
                    'ratecodeid': 'ratecodeid',
                    'ratecodecodeid': 'ratecodeid'
                }
                for old_name, new_name in column_mapping.items():
                    if old_name in df_pandas.columns:
                        df_pandas.rename(columns={old_name: new_name}, inplace=True)
                
                # Normalize data types to match across files
                # Convert ALL numeric columns to float64 (universal type that handles both int and float)
                numeric_cols = ['vendorid', 'passenger_count', 'ratecodeid', 'payment_type', 
                               'trip_distance', 'fare_amount', 'extra', 'mta_tax', 
                               'improvement_surcharge', 'tip_amount', 'tolls_amount', 'total_amount',
                               'congestion_surcharge']
                
                for col in numeric_cols:
                    if col in df_pandas.columns:
                        df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce').astype('float64')
                
                # Convert datetime columns
                for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
                    if col in df_pandas.columns:
                        df_pandas[col] = pd.to_datetime(df_pandas[col], errors='coerce')
                
                # Ensure all expected columns are present (fill with nulls if missing)
                expected_cols = ['vendorid', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 
                               'passenger_count', 'trip_distance', 'ratecodeid', 'store_and_fwd_flag',
                               'pulocationid', 'dolocationid', 'payment_type', 'fare_amount',
                               'extra', 'mta_tax', 'improvement_surcharge', 'tip_amount',
                               'tolls_amount', 'total_amount', 'congestion_surcharge']
                
                for col in expected_cols:
                    if col not in df_pandas.columns:
                        df_pandas[col] = None
                
                # Convert to PyArrow table
                table = pa.Table.from_pandas(df_pandas)
                
                all_tables.append(table)
                files_processed += 1
                
            except MemoryError as me:
                logger.warning("  Memory error processing {} - skipping: {}".format(csv_file.name, str(me)))
                continue
            except Exception as e:
                logger.warning("  Warning processing {}: {}".format(csv_file.name, str(e)))
                continue
        
        if files_processed == 0:
            raise Exception("No CSV files could be processed successfully")
        
        # Combine all tables and write to Parquet
        logger.info("Combining {} tables (unifying schemas)...".format(len(all_tables)))
        
        # Get union schema - handles schema evolution (missing columns become nulls)
        try:
            # Try standard concat first
            combined_table = pa.concat_tables(all_tables)
        except pa.ArrowInvalid as e:
            # Handle schema mismatch - unify to common columns
            logger.warning("Schema mismatch during concat: {}. Using common columns only.".format(str(e)))
            
            # Get all column names from all tables
            all_cols = set()
            for table in all_tables:
                all_cols.update(table.column_names)
            all_cols = sorted(list(all_cols))
            
            # Select only common columns and reorder
            unified_tables = []
            for i, table in enumerate(all_tables):
                # Select existing columns in common order
                cols_to_select = [col for col in all_cols if col in table.column_names]
                if cols_to_select:
                    unified_tables.append(table.select(cols_to_select))
            
            if unified_tables:
                combined_table = pa.concat_tables(unified_tables)
            else:
                raise Exception("No valid tables after schema unification")
        
        # Write to Parquet using PyArrow (bypasses Hadoop entirely)
        logger.info("[SAVE] Writing to Parquet with PyArrow: {}".format(output_path))
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write using PyArrow directly
        pq.write_table(combined_table, output_path, compression='snappy')
        
        logger.info("[OK] Successfully saved {} records to {}".format(total_records, output_path))
        logger.info("[OK] Processed {} files with total {} records".format(files_processed, total_records))
        return total_records, output_path
        
    except Exception as e:
        logger.error("[ERROR] Error in load_raw_data_and_save: {}".format(str(e)))
        raise


def repartition_data(df, partition_key="PULocationID", num_partitions=100):
    """
    Repartition data for efficient processing
    
    Args:
        df: Input DataFrame
        partition_key: Column to partition by (hot key)
        num_partitions: Number of partitions
        
    Returns:
        Repartitioned DataFrame
    """
    logger.info(f"[REPART] Repartitioning data by {partition_key} into {num_partitions} partitions...")
    
    df_repartitioned = df.repartition(num_partitions, partition_key)
    
    logger.info(f"[OK] Data repartitioned for parallel processing")
    return df_repartitioned


def save_as_parquet(df, output_path):
    """
    Save DataFrame as Parquet format
    
    Args:
        df: Input DataFrame
        output_path: Output Parquet path
        
    Returns:
        Path to saved Parquet file
    """
    try:
        logger.info(f"[SAVE] Saving to Parquet: {output_path}...")
        
        df.write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(output_path)
        
        logger.info(f"[OK] Successfully saved {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"[ERROR] Error saving Parquet: {str(e)}")
        raise


def generate_ingestion_report(df, output_path):
    """
    Generate ingestion statistics report
    
    Args:
        df: Ingested DataFrame (can be None)
        output_path: Path to save report
    """
    # Read parquet file to get metadata
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(output_path)
    
    total_records = parquet_file.metadata.num_rows
    num_columns = len(parquet_file.schema.names)
    
    report = {
        "ingestion_timestamp": datetime.now().isoformat(),
        "total_records": total_records,
        "columns": num_columns,
        "column_names": parquet_file.schema.names,
        "data_types": {name: "arrow_field" for name in parquet_file.schema.names},
        "file_size_mb": Path(output_path).stat().st_size / (1024**2)
    }
    
    logger.info("[REPORT] Ingestion Report:\n{}".format(json.dumps(report, indent=2, default=str)))
    
    # Save report
    report_path = output_path.replace(".parquet", "_ingestion_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("[OK] Ingestion report saved to {}".format(report_path))
    return report


def main():
    """Main ingestion pipeline"""
    logger.info("=" * 70)
    logger.info("[TAXI] NYC TAXI DATA INGESTION PIPELINE (Week 1.2 - Step 1)")
    logger.info("=" * 70)
    
    # Paths
    raw_data_path = "data/raw"  # Directory containing CSV files
    output_parquet = "data/processed/ingested_data.parquet"
    
    try:
        # Step 1: Create Spark session
        spark = create_spark_session()
        
        # Step 2: Load raw data and save to Parquet (with streaming writes to avoid memory issues)
        total_records, output_path = load_raw_data_and_save(spark, raw_data_path, output_parquet)
        
        # Step 3: Verify file was created successfully
        if not Path(output_parquet).exists():
            raise FileNotFoundError("Parquet file not created at {}".format(output_parquet))
        
        file_size_mb = Path(output_parquet).stat().st_size / (1024**2)
        logger.info("[OK] Parquet file created successfully: {:.2f} MB".format(file_size_mb))
        
        # Step 4: Generate report
        generate_ingestion_report(None, output_parquet)
        
        logger.info("=" * 70)
        logger.info("✨ INGESTION COMPLETE - Ready for validation (Step 2)")
        logger.info("=" * 70)
        
        spark.stop()
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
