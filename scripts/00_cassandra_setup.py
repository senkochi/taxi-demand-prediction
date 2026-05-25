"""
Task 1.5: Cassandra Schema Setup & Validation
Purpose: Create keyspace, tables, and prepare for feature ingestion
Execution: python scripts/00_cassandra_setup.py
"""

import time
import logging
import os
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

# ============================================================================
# LOGGING
# ============================================================================

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("CassandraSetup")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("logs/00_cassandra_setup.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)
logger.addHandler(file_handler)

# ============================================================================
# CASSANDRA SETUP
# ============================================================================

class CassandraSetup:
    def __init__(self, contact_points=['127.0.0.1', '127.0.0.2', '127.0.0.3'], 
                 port=9042, username='cassandra', password='cassandra'):
        self.contact_points = contact_points
        self.port = port
        self.username = username
        self.password = password
        self.cluster = None
        self.session = None

    def connect(self):
        """Connect to Cassandra cluster"""
        logger.info(f"🔗 Connecting to Cassandra cluster: {self.contact_points}:{self.port}...")
        
        try:
            self.cluster = Cluster(
                contact_points=self.contact_points,
                port=self.port,
                auth_provider=None,  # Change if auth required
                load_balancing_policy=RoundRobinPolicy()
            )
            self.session = self.cluster.connect()
            logger.info("✅ Connected to Cassandra cluster")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect: {str(e)}")
            return False

    def wait_for_cluster(self, max_retries=30):
        """Wait for Cassandra to be ready"""
        logger.info("⏳ Waiting for Cassandra cluster to be ready...")
        
        for attempt in range(max_retries):
            try:
                # Simple health check
                result = self.session.execute("SELECT release_version FROM system.local")
                version = result[0].release_version
                logger.info(f"✅ Cassandra {version} is ready (attempt {attempt+1})")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"  ⏳ Attempt {attempt+1}/{max_retries} - Waiting... ({str(e)[:50]})")
                    time.sleep(2)
                else:
                    logger.error(f"❌ Cassandra not ready after {max_retries} attempts")
                    return False

    def create_keyspace(self):
        """Create keyspace with replication factor 3"""
        logger.info("📦 Creating keyspace 'taxi_db'...")
        
        try:
            cql = """
            CREATE KEYSPACE IF NOT EXISTS taxi_db
            WITH replication = {
                'class': 'SimpleStrategy',
                'replication_factor': 3
            }
            AND durable_writes = true;
            """
            self.session.execute(cql)
            logger.info("✅ Keyspace 'taxi_db' created")
            
            # Set keyspace for subsequent operations
            self.session.set_keyspace('taxi_db')
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create keyspace: {str(e)}")
            return False

    def create_tables(self):
        """Create all required tables"""
        logger.info("🗂️  Creating tables...")
        
        tables = {
            "baseline_features": """
                CREATE TABLE IF NOT EXISTS baseline_features (
                    zone_id int,
                    window_start timestamp,
                    date_str date,
                    date_day int,
                    hour int,
                    demand_count bigint,
                    avg_fare float,
                    sum_fare float,
                    min_fare float,
                    max_fare float,
                    stddev_fare float,
                    avg_distance float,
                    median_distance float,
                    p95_distance float,
                    avg_passenger float,
                    PRIMARY KEY (zone_id, window_start)
                )
                WITH CLUSTERING ORDER BY (window_start DESC)
                AND default_time_to_live = 0
                AND comment = 'Baseline taxi demand features (15-min aggregation)';
            """,
            
            "method1_clusters": """
                CREATE TABLE IF NOT EXISTS method1_clusters (
                    zone_id int PRIMARY KEY,
                    cluster_id int,
                    demand_mean float,
                    demand_std float,
                    demand_peak_hour int,
                    night_demand_ratio float,
                    morning_peak float,
                    evening_peak float,
                    created_at timestamp
                )
                WITH comment = 'Method 1: Demand-based clustering assignments';
            """,
            
            "method2_clusters": """
                CREATE TABLE IF NOT EXISTS method2_clusters (
                    zone_id int PRIMARY KEY,
                    cluster_id int,
                    trip_count bigint,
                    avg_trip_distance float,
                    avg_fare float,
                    avg_passenger_count float,
                    avg_trip_duration bigint,
                    created_at timestamp
                )
                WITH comment = 'Method 2: Mobility pattern clustering';
            """,
            
            "method3_clusters": """
                CREATE TABLE IF NOT EXISTS method3_clusters (
                    zone_id int PRIMARY KEY,
                    cluster_id int,
                    inflow bigint,
                    outflow bigint,
                    trip_count bigint,
                    avg_distance float,
                    avg_fare float,
                    created_at timestamp
                )
                WITH comment = 'Method 3: OD-flow based clustering';
            """,
            
            "cluster_metadata": """
                CREATE TABLE IF NOT EXISTS cluster_metadata (
                    method text,
                    cluster_id int,
                    num_zones int,
                    silhouette_score float,
                    interpretation text,
                    created_at timestamp,
                    PRIMARY KEY (method, cluster_id)
                )
                WITH comment = 'Clustering metadata and interpretations';
            """
        }
        
        for table_name, cql in tables.items():
            try:
                self.session.execute(cql)
                logger.info(f"  ✅ Table '{table_name}' created")
            except Exception as e:
                logger.error(f"  ❌ Failed to create table '{table_name}': {str(e)}")
                return False
        
        logger.info("✅ All tables created")
        return True

    def create_indexes(self):
        """Create indexes for common queries"""
        logger.info("🔍 Creating indexes...")
        
        indexes = {
            "baseline_features_date_day": """
                CREATE INDEX IF NOT EXISTS baseline_features_date_day 
                ON baseline_features (date_day);
            """,
            "baseline_features_hour": """
                CREATE INDEX IF NOT EXISTS baseline_features_hour 
                ON baseline_features (hour);
            """,
            "method1_cluster_id": """
                CREATE INDEX IF NOT EXISTS method1_cluster_id 
                ON method1_clusters (cluster_id);
            """
        }
        
        for index_name, cql in indexes.items():
            try:
                self.session.execute(cql)
                logger.info(f"  ✅ Index '{index_name}' created")
            except Exception as e:
                logger.warning(f"  ⚠️  Index '{index_name}' creation: {str(e)[:80]}")
        
        logger.info("✅ Indexes created")
        return True

    def verify_schema(self):
        """Verify schema was created correctly"""
        logger.info("✅ Verifying schema...")
        
        try:
            # List tables
            tables = self.session.cluster.metadata.keyspaces['taxi_db'].tables.keys()
            logger.info(f"  Tables: {', '.join(sorted(tables))}")
            
            # Describe each table
            for table_name in sorted(tables):
                table = self.session.cluster.metadata.keyspaces['taxi_db'].tables[table_name]
                col_count = len(table.columns)
                logger.info(f"    - {table_name}: {col_count} columns")
            
            logger.info("✅ Schema verification complete")
            return True
        except Exception as e:
            logger.error(f"❌ Schema verification failed: {str(e)}")
            return False

    def disconnect(self):
        """Close connection"""
        if self.session:
            self.session.shutdown()
        if self.cluster:
            self.cluster.shutdown()
        logger.info("🔌 Disconnected from Cassandra")

    def setup(self):
        """Execute full setup"""
        logger.info("=" * 70)
        logger.info("🚕 CASSANDRA CLUSTER SETUP (Week 1.1, Task 1.5)")
        logger.info("=" * 70)
        
        try:
            # Connect
            if not self.connect():
                return 1
            
            # Wait for cluster
            if not self.wait_for_cluster():
                return 1
            
            # Create keyspace
            if not self.create_keyspace():
                return 1
            
            # Create tables
            if not self.create_tables():
                return 1
            
            # Create indexes
            if not self.create_indexes():
                return 1
            
            # Verify
            if not self.verify_schema():
                return 1
            
            logger.info("=" * 70)
            logger.info("✅ CASSANDRA SETUP COMPLETE")
            logger.info("=" * 70)
            logger.info("Ready to load baseline_features.parquet into Cassandra")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
        
        finally:
            self.disconnect()

# ============================================================================
# MAIN
# ============================================================================

def main():
    setup = CassandraSetup(
        contact_points=['127.0.0.1', '127.0.0.2', '127.0.0.3'],
        port=9042
    )
    exit_code = setup.setup()
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
