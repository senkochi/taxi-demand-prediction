"""
Enrich baseline_features with taxi zone lookup metadata
Adds Borough, Zone, and service_zone columns
"""
import duckdb
import pandas as pd
from pathlib import Path
import json

db_path = Path("data/processed/taxi_features.duckdb")
zone_csv = Path("data/raw/taxi+_zone_lookup.csv")
report_file = Path("logs/04_zone_enrichment_report.json")

print("=" * 80)
print("ENRICHING BASELINE FEATURES WITH ZONE LOOKUP")
print("=" * 80)

# Connect to DuckDB
conn = duckdb.connect(str(db_path))

# Step 1: Load zone lookup CSV
print("\n[1/4] Loading zone lookup CSV...")
try:
    zone_df = pd.read_csv(zone_csv)
    print(f"✅ Loaded {len(zone_df)} zone records")
    print(f"   Columns: {list(zone_df.columns)}")
except Exception as e:
    print(f"❌ Error loading zone CSV: {e}")
    exit(1)

# Step 2: Get current baseline_features count
print("\n[2/4] Checking baseline_features table...")
current_count = conn.execute("SELECT COUNT(*) FROM baseline_features").fetchone()[0]
print(f"✅ Current records: {current_count:,}")

# Step 3: Create enriched view with zone lookup
print("\n[3/4] Creating enriched dataset...")
try:
    # Register zone_df as a table in DuckDB
    conn.register('zone_lookup', zone_df)
    
    # Create enriched features with zone metadata
    enriched_sql = """
    SELECT 
        bf.*,
        zl.Borough,
        zl.Zone,
        zl.service_zone
    FROM baseline_features bf
    LEFT JOIN zone_lookup zl ON bf.zone_id = zl.LocationID
    """
    
    enriched_df = conn.execute(enriched_sql).df()
    print(f"✅ Created enriched dataset: {len(enriched_df):,} rows × {len(enriched_df.columns)} columns")
    print(f"   New columns: Borough, Zone, service_zone")
    
    # Check for any nulls in zone columns
    null_zones = enriched_df[enriched_df['zone_id'].notna() & enriched_df['Borough'].isna()].shape[0]
    if null_zones > 0:
        print(f"⚠️ Warning: {null_zones} zones with missing Borough info")
    else:
        print(f"✅ All zones have matching Borough information")
        
except Exception as e:
    print(f"❌ Error creating enriched dataset: {e}")
    exit(1)

# Step 4: Replace baseline_features table with enriched version
print("\n[4/4] Updating database...")
try:
    # Drop old table
    conn.execute("DROP TABLE IF EXISTS baseline_features")
    
    # Create new table with enriched data
    conn.execute(f"CREATE TABLE baseline_features AS SELECT * FROM enriched_df")
    
    # Create indexes for common queries
    conn.execute("CREATE INDEX idx_zone_borough ON baseline_features(zone_id, Borough)")
    conn.execute("CREATE INDEX idx_borough_time ON baseline_features(Borough, window_start)")
    conn.execute("CREATE INDEX idx_service_zone ON baseline_features(service_zone)")
    
    print("✅ Table updated with zone metadata")
    print("✅ Indexes created for: zone+borough, borough+time, service_zone")
    
except Exception as e:
    print(f"❌ Error updating table: {e}")
    exit(1)

# Step 5: Verification
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

# Check schema
print("\n📋 New Schema:")
schema = conn.execute("PRAGMA table_info(baseline_features)").fetchall()
for col in schema:
    print(f"  {col[1]:25s} {col[2]:15s}")

# Sample with borough
print("\n📄 Sample Data (with zone info):")
sample = conn.execute("""
    SELECT zone_id, Borough, Zone, service_zone, demand_count, window_start
    FROM baseline_features 
    LIMIT 5
""").fetchall()
for row in sample:
    print(f"  Zone {row[0]:3d}: {row[1]:15s} | {row[2]:35s} | {row[3]:20s}")

# Borough distribution
print("\n🗺️ Zone Distribution by Borough:")
borough_stats = conn.execute("""
    SELECT 
        Borough,
        COUNT(DISTINCT zone_id) as num_zones,
        COUNT(*) as total_records,
        AVG(demand_count) as avg_demand
    FROM baseline_features
    GROUP BY Borough
    ORDER BY total_records DESC
""").fetchall()

for borough, num_zones, total_records, avg_demand in borough_stats:
    print(f"  {borough:20s}: {num_zones:3d} zones | {total_records:10,d} records | avg_demand: {avg_demand:6.1f}")

# Service zone distribution
print("\n📍 Distribution by Service Zone:")
service_stats = conn.execute("""
    SELECT 
        service_zone,
        COUNT(DISTINCT zone_id) as num_zones,
        COUNT(*) as total_records
    FROM baseline_features
    GROUP BY service_zone
    ORDER BY total_records DESC
""").fetchall()

for service_zone, num_zones, total_records in service_stats:
    print(f"  {service_zone:20s}: {num_zones:3d} zones | {total_records:10,d} records")

# Generate report
report = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "operation": "zone_enrichment",
    "status": "completed",
    "original_records": current_count,
    "enriched_records": len(enriched_df),
    "columns_added": ["Borough", "Zone", "service_zone"],
    "borough_distribution": {
        row[0]: {"zones": row[1], "records": row[2], "avg_demand": float(row[3])}
        for row in borough_stats
    },
    "service_zone_distribution": {
        row[0]: {"zones": row[1], "records": row[2]}
        for row in service_stats
    },
    "data_quality": {
        "total_rows": len(enriched_df),
        "zones_without_borough": int(null_zones),
        "coverage_pct": 100.0 * (len(enriched_df) - null_zones) / len(enriched_df) if len(enriched_df) > 0 else 0
    }
}

# Save report
report_file.parent.mkdir(parents=True, exist_ok=True)
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Report saved to {report_file}")

# Cleanup
conn.close()

print("\n" + "=" * 80)
print("✅ ENRICHMENT COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"  • {len(enriched_df):,} baseline features enriched with zone metadata")
print(f"  • Boroughs represented: {len(borough_stats)}")
print(f"  • Service zones: {len(service_stats)}")
print(f"  • Ready for Method 1/2/3 clustering analysis")
print(f"\n🔗 Next steps:")
print(f"  1. Query by Borough: SELECT * FROM baseline_features WHERE Borough='Manhattan'")
print(f"  2. Use for clustering: GROUP BY Borough, Zone, service_zone")
print(f"  3. Visualize patterns by geography")
print("\n")
