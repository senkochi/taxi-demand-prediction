"""
Explore DuckDB database structure and sample data
"""
import duckdb
import json
from pathlib import Path

db_path = Path("data/processed/taxi_features.duckdb")
zone_csv = Path("data/raw/taxi_zone_lookup.csv")

print("=" * 80)
print("DUCKDB STRUCTURE EXPLORATION")
print("=" * 80)

# Connect to DuckDB
conn = duckdb.connect(str(db_path), read_only=True)

# List all tables
print("\n📊 TABLES IN DATABASE:")
tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
for table in tables:
    print(f"  - {table[0]}")

# Schema for baseline_features
print("\n📋 SCHEMA: baseline_features")
schema = conn.execute("PRAGMA table_info(baseline_features)").fetchall()
for col in schema:
    print(f"  {col[1]:20s} {col[2]:15s}")

# Sample data
print("\n📄 SAMPLE DATA (first 3 rows):")
sample = conn.execute("SELECT * FROM baseline_features LIMIT 3").fetchall()
print(f"Columns: {[desc[0] for desc in conn.description]}")
for row in sample:
    print(f"  {row}")

# Data summary
print("\n📈 DATA SUMMARY:")
summary = conn.execute("""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT zone_id) as unique_zones,
        MIN(window_start) as earliest_time,
        MAX(window_start) as latest_time,
        AVG(demand_count) as avg_demand
    FROM baseline_features
""").fetchone()
print(f"  Total rows: {summary[0]:,}")
print(f"  Unique zones: {summary[1]}")
print(f"  Time range: {summary[2]} to {summary[3]}")
print(f"  Avg demand: {summary[4]:.1f}")

# Check zone values
print("\n🗺️ UNIQUE ZONES IN DATABASE:")
zones = conn.execute("SELECT DISTINCT zone_id FROM baseline_features ORDER BY zone_id LIMIT 20").fetchall()
print(f"  First 20 zones: {[z[0] for z in zones]}")
print(f"  ...")

conn.close()

# Check if taxi_zone_lookup.csv exists
print("\n" + "=" * 80)
print("TAXI ZONE LOOKUP CSV")
print("=" * 80)

if zone_csv.exists():
    print(f"✅ File exists: {zone_csv}")
    print(f"\n📄 SAMPLE from CSV (first 5 rows):")
    with open(zone_csv, 'r') as f:
        for i, line in enumerate(f):
            if i < 6:  # header + 5 data rows
                print(f"  {line.strip()}")
            else:
                break
else:
    print(f"❌ File NOT found: {zone_csv}")

print("\n" + "=" * 80)
print("STATUS: Is zone_lookup implemented?")
print("=" * 80)

# Try loading zone lookup with DuckDB
try:
    zone_df = duckdb.read_csv(str(zone_csv)).df()
    print(f"✅ Zone lookup CSV can be read")
    print(f"   Columns: {list(zone_df.columns)}")
    print(f"   Rows: {len(zone_df)}")
    print(f"\n   Sample:")
    print(zone_df.head(3))
except Exception as e:
    print(f"❌ Error reading zone lookup: {e}")

print("\n" + "=" * 80)
