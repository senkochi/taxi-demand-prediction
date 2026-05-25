#!/usr/bin/env python3
"""
Test queries on DuckDB to verify baseline features are loaded correctly.
"""

import duckdb
import json
from pathlib import Path

DB_PATH = "data/processed/taxi_features.duckdb"

def test_queries():
    """Run test queries on DuckDB"""
    
    # Connect to database
    conn = duckdb.connect(DB_PATH, read_only=True)
    
    print("=" * 80)
    print("🔍 DUCKDB BASELINE FEATURES - TEST QUERIES")
    print("=" * 80)
    
    # Query 1: Table structure
    print("\n1️⃣ TABLE SCHEMA:")
    schema = conn.execute("DESCRIBE baseline_features").fetchall()
    for row in schema:
        print(f"  {row[0]:25} | {row[1]}")
    
    # Query 2: Record count & date range
    print("\n2️⃣ DATASET STATISTICS:")
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT zone_id) as unique_zones,
            MIN(window_start) as earliest_window,
            MAX(window_start) as latest_window,
            MIN(date_str) as earliest_date,
            MAX(date_str) as latest_date
        FROM baseline_features
    """).fetchall()
    row = stats[0]
    print(f"  Total Records:     {row[0]:,}")
    print(f"  Unique Zones:      {row[1]}")
    print(f"  Earliest Window:   {row[2]}")
    print(f"  Latest Window:     {row[3]}")
    print(f"  Date Range:        {row[4]} to {row[5]}")
    
    # Query 3: Top 5 zones by demand
    print("\n3️⃣ TOP 5 ZONES BY TOTAL DEMAND:")
    top_zones = conn.execute("""
        SELECT 
            zone_id,
            SUM(demand_count) as total_demand,
            COUNT(*) as time_windows,
            AVG(demand_count) as avg_demand,
            MAX(demand_count) as peak_demand
        FROM baseline_features
        GROUP BY zone_id
        ORDER BY total_demand DESC
        LIMIT 5
    """).fetchall()
    for row in top_zones:
        print(f"  Zone {row[0]:3} | Total: {row[1]:8,} | Windows: {row[2]:5} | Avg: {row[3]:7.1f} | Peak: {row[4]:6.0f}")
    
    # Query 4: Hourly demand pattern (average across all zones)
    print("\n4️⃣ AVERAGE HOURLY DEMAND PATTERN (all zones combined):")
    hourly = conn.execute("""
        SELECT 
            hour,
            COUNT(*) as num_windows,
            AVG(demand_count) as avg_demand,
            MAX(demand_count) as peak_demand,
            MIN(demand_count) as min_demand
        FROM baseline_features
        GROUP BY hour
        ORDER BY hour
    """).fetchall()
    for row in hourly:
        bar = "█" * int(row[2] / 50)
        print(f"  Hour {row[0]:2}h | Windows: {row[1]:5} | Avg: {row[2]:7.1f} | Peak: {row[3]:6.0f} | Min: {row[4]:6.0f} | {bar}")
    
    # Query 5: Data quality check (nulls, missing values)
    print("\n5️⃣ DATA QUALITY CHECK:")
    quality = conn.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN zone_id IS NULL THEN 1 END) as null_zone_id,
            COUNT(CASE WHEN window_start IS NULL THEN 1 END) as null_window_start,
            COUNT(CASE WHEN demand_count IS NULL THEN 1 END) as null_demand_count,
            COUNT(CASE WHEN demand_count < 0 THEN 1 END) as negative_demand,
            COUNT(CASE WHEN demand_count = 0 THEN 1 END) as zero_demand
        FROM baseline_features
    """).fetchall()
    row = quality[0]
    print(f"  Total Records:        {row[0]:,}")
    print(f"  Null zone_id:         {row[1]}")
    print(f"  Null window_start:    {row[2]}")
    print(f"  Null demand_count:    {row[3]}")
    print(f"  Negative demand:      {row[4]}")
    print(f"  Zero demand:          {row[5]}")
    
    # Query 6: Fare statistics
    print("\n6️⃣ FARE STATISTICS (aggregated):")
    fare_stats = conn.execute("""
        SELECT 
            AVG(avg_fare) as mean_avg_fare,
            MAX(max_fare) as max_fare_observed,
            MIN(min_fare) as min_fare_observed,
            AVG(sum_fare) as mean_sum_fare,
            AVG(stddev_fare) as mean_stddev_fare
        FROM baseline_features
    """).fetchall()
    row = fare_stats[0]
    print(f"  Mean Avg Fare:        ${row[0]:.2f}")
    print(f"  Max Fare Observed:    ${row[1]:.2f}")
    print(f"  Min Fare Observed:    ${row[2]:.2f}")
    print(f"  Mean Sum Fare:        ${row[3]:.2f}")
    print(f"  Mean StdDev Fare:     ${row[4]:.2f}")
    
    # Query 7: Distance statistics
    print("\n7️⃣ DISTANCE STATISTICS (aggregated):")
    dist_stats = conn.execute("""
        SELECT 
            AVG(avg_distance) as mean_avg_distance,
            AVG(median_distance) as mean_median_distance,
            AVG(p95_distance) as mean_p95_distance,
            MAX(p95_distance) as max_p95_distance
        FROM baseline_features
    """).fetchall()
    row = dist_stats[0]
    print(f"  Mean Avg Distance:    {row[0]:.2f} miles")
    print(f"  Mean Median Distance: {row[1]:.2f} miles")
    print(f"  Mean P95 Distance:    {row[2]:.2f} miles")
    print(f"  Max P95 Distance:     {row[3]:.2f} miles")
    
    # Query 8: Sample row (for inspection)
    print("\n8️⃣ SAMPLE ROW (inspection):")
    sample = conn.execute("""
        SELECT * FROM baseline_features 
        LIMIT 1
    """).fetchall()
    sample_dict = dict(zip([desc[0] for desc in conn.description], sample[0]))
    for key, value in sample_dict.items():
        print(f"  {key:20} : {value}")
    
    # Query 9: Time window distribution
    print("\n9️⃣ TIME WINDOW DISTRIBUTION:")
    window_dist = conn.execute("""
        SELECT 
            time_bucket,
            COUNT(*) as num_zones,
            AVG(demand_count) as avg_demand
        FROM baseline_features
        GROUP BY time_bucket
        ORDER BY time_bucket
    """).fetchall()
    print(f"  Total unique time windows: {len(window_dist)}")
    print(f"  Sample windows:")
    for row in window_dist[:5]:
        print(f"    {row[0]:20} | Zones: {row[1]:3} | Avg Demand: {row[2]:.1f}")
    
    # Query 10: Indexes info
    print("\n🔟 INDEXES AVAILABLE:")
    try:
        indexes = conn.execute("SELECT * FROM duckdb_indexes()").fetchall()
        if indexes:
            for row in indexes:
                print(f"  {row}")
        else:
            print("  No indexes defined (will create during clustering)")
    except:
        print("  Indexes info not available")
    
    print("\n" + "=" * 80)
    print("✅ DUCKDB DATABASE IS OPERATIONAL & READY FOR CLUSTERING")
    print("=" * 80)
    
    conn.close()

if __name__ == "__main__":
    test_queries()
