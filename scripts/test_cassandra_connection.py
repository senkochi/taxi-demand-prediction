#!/usr/bin/env python
"""
Quick Cassandra connection test
"""
import sys
import time
import socket
from cassandra.cluster import Cluster

def test_port_open(host, port):
    """Check if port is open"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def test_cassandra_connection():
    """Test Cassandra connection"""
    print("=" * 70)
    print("🧪 CASSANDRA CONNECTION TEST")
    print("=" * 70)
    
    # Step 1: Check if ports are accessible
    print("\n1️⃣  Checking if Cassandra ports are accessible...")
    ports = [9042, 9043, 9044]
    for port in ports:
        open = test_port_open('127.0.0.1', port)
        status = "✅ OPEN" if open else "❌ CLOSED"
        print(f"   Port {port}: {status}")
    
    # Step 2: Try to connect
    print("\n2️⃣  Attempting to connect to Cassandra...")
    try:
        cluster = Cluster(['127.0.0.1'], port=9042)
        session = cluster.connect('taxi_db')
        print("   ✅ Connected successfully!")
        
        # Step 3: Test query
        print("\n3️⃣  Testing SELECT query...")
        result = session.execute("SELECT COUNT(*) FROM baseline_features")
        count = result[0].count
        print(f"   ✅ Query succeeded - {count:,} rows in baseline_features")
        
        # Step 4: Test insert
        print("\n4️⃣  Testing INSERT statement...")
        insert_stmt = session.prepare("""
            INSERT INTO baseline_features 
            (zone_id, window_start, date_str, date_day, hour, demand_count, 
             avg_fare, sum_fare, min_fare, max_fare, stddev_fare, 
             avg_distance, median_distance, p95_distance, avg_passenger)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        print("   ✅ Prepared statement created successfully")
        
        session.shutdown()
        cluster.shutdown()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED - Cassandra is ready!")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("❌ TESTS FAILED - Cassandra is not accessible")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit_code = test_cassandra_connection()
    sys.exit(exit_code)
