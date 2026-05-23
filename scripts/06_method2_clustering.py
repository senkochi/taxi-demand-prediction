"""
Method 2: Mobility Pattern Clustering ⭐ (Primary Method)
Cluster zones by movement patterns: trip distance, fare, passengers, volume
"""
import duckdb
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
db_path = Path("data/processed/taxi_features.duckdb")
output_dir = Path("data/processed/method2_features")
models_dir = Path("data/models")
report_dir = Path("logs")
figures_dir = Path("reports/figures")

# Create directories
output_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("METHOD 2: MOBILITY PATTERN CLUSTERING ⭐ (PRIMARY)")
print("=" * 80)

# Connect to DuckDB
conn = duckdb.connect(str(db_path), read_only=True)

# Step 1: Extract per-zone mobility aggregations
print("\n[1/6] Extracting per-zone mobility patterns...")
try:
    # Aggregate across all time periods per zone
    mobility_sql = """
    SELECT 
        zone_id,
        Borough,
        Zone,
        service_zone,
        SUM(demand_count) as trip_count,
        AVG(avg_distance) as avg_trip_distance,
        SUM(sum_fare) / SUM(demand_count) as avg_fare,
        AVG(avg_passenger) as avg_passenger_count,
        STDDEV(demand_count) as demand_variability,
        MAX(demand_count) as peak_demand,
        COUNT(*) as num_time_buckets
    FROM baseline_features
    GROUP BY zone_id, Borough, Zone, service_zone
    ORDER BY trip_count DESC
    """
    
    mobility_df = conn.execute(mobility_sql).df()
    print(f"✅ Extracted mobility features for {len(mobility_df)} zones")
    print(f"\n   Feature statistics:")
    print(f"     trip_count:          {mobility_df['trip_count'].mean():,.0f} ± {mobility_df['trip_count'].std():,.0f}")
    print(f"     avg_trip_distance:   {mobility_df['avg_trip_distance'].mean():.2f} ± {mobility_df['avg_trip_distance'].std():.2f} miles")
    print(f"     avg_fare:            ${mobility_df['avg_fare'].mean():.2f} ± ${mobility_df['avg_fare'].std():.2f}")
    print(f"     avg_passenger_count: {mobility_df['avg_passenger_count'].mean():.2f} ± {mobility_df['avg_passenger_count'].std():.2f}")
    
    # Compute trip_duration proxy (based on distance: ~2.5 minutes per mile in NYC)
    mobility_df['avg_trip_duration_min'] = mobility_df['avg_trip_distance'] * 2.5
    
except Exception as e:
    print(f"❌ Error extracting mobility features: {e}")
    exit(1)

# Step 2: Feature normalization
print("\n[2/6] Normalizing features for clustering...")
try:
    # Select features for clustering
    clustering_cols = [
        'trip_count', 'avg_trip_distance', 'avg_fare', 
        'avg_passenger_count', 'avg_trip_duration_min', 'demand_variability'
    ]
    
    clustering_features = mobility_df[clustering_cols].copy()
    
    # Handle NaN values
    clustering_features = clustering_features.fillna(clustering_features.mean())
    
    # Log-transform trip_count to reduce skewness (some zones have 1000x more trips)
    clustering_features['trip_count'] = np.log1p(clustering_features['trip_count'])
    
    # Standardize (zero mean, unit variance)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_features)
    
    print(f"✅ Normalized {scaled_features.shape[1]} features")
    print(f"   Shape: {scaled_features.shape[0]} zones × {scaled_features.shape[1]} features")
    print(f"\n   Features used:")
    for i, col in enumerate(clustering_cols):
        print(f"     {i+1}. {col}")
    
except Exception as e:
    print(f"❌ Error normalizing features: {e}")
    exit(1)

# Step 3: K-Means clustering with silhouette and Davies-Bouldin analysis
print("\n[3/6] Running K-Means with multi-metric analysis...")
try:
    silhouette_scores = {}
    davies_bouldin_scores = {}
    models = {}
    predictions_by_k = {}
    inertias = {}
    
    k_range = range(3, 9)  # Try K from 3 to 8
    
    for k in k_range:
        print(f"\n  K={k}...", end=" ", flush=True)
        
        # Train K-Means
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            max_iter=300,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(scaled_features)
        
        # Compute metrics
        sil_score = silhouette_score(scaled_features, labels)
        db_score = davies_bouldin_score(scaled_features, labels)
        inertia = kmeans.inertia_
        
        silhouette_scores[k] = sil_score
        davies_bouldin_scores[k] = db_score
        models[k] = kmeans
        predictions_by_k[k] = labels
        inertias[k] = inertia
        
        print(f"Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}, Inertia={inertia:.1f}")
    
    # Find optimal K (highest silhouette, lower davies-bouldin is better)
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    optimal_model = models[optimal_k]
    optimal_labels = predictions_by_k[optimal_k]
    
    print(f"\n✅ Clustering analysis complete")
    print(f"   Optimal K: {optimal_k}")
    print(f"   Silhouette score: {silhouette_scores[optimal_k]:.4f}")
    print(f"   Davies-Bouldin score: {davies_bouldin_scores[optimal_k]:.4f} (lower is better)")
    
    # Cluster distribution
    print(f"\n   Cluster distribution (K={optimal_k}):")
    unique, counts = np.unique(optimal_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = 100.0 * count / len(optimal_labels)
        print(f"     Cluster {cluster_id}: {count:3d} zones ({pct:5.1f}%)")
    
except Exception as e:
    print(f"❌ Error in K-Means clustering: {e}")
    exit(1)

# Step 4: Cluster interpretation & semantics
print("\n" + "=" * 80)
print("CLUSTER ANALYSIS & INTERPRETATION")
print("=" * 80)

mobility_df['cluster'] = optimal_labels

# Zone to cluster mapping
zone_to_cluster = dict(zip(mobility_df['zone_id'], mobility_df['cluster']))

# Analyze each cluster
cluster_analysis = {}
cluster_semantics = {}

for cluster_id in range(optimal_k):
    cluster_data = mobility_df[mobility_df['cluster'] == cluster_id]
    
    # Compute statistics
    analysis = {
        'num_zones': len(cluster_data),
        'boroughs': cluster_data['Borough'].value_counts().to_dict(),
        'avg_trip_count': float(cluster_data['trip_count'].mean()),
        'avg_trip_distance': float(cluster_data['avg_trip_distance'].mean()),
        'avg_fare': float(cluster_data['avg_fare'].mean()),
        'avg_passenger_count': float(cluster_data['avg_passenger_count'].mean()),
        'avg_demand_variability': float(cluster_data['demand_variability'].mean()),
        'top_zones': cluster_data.nlargest(3, 'trip_count')[['zone_id', 'Zone', 'trip_count']].to_dict('records')
    }
    
    cluster_analysis[int(cluster_id)] = analysis
    
    # Assign semantics based on characteristics
    avg_distance = analysis['avg_trip_distance']
    avg_trip_cnt = analysis['avg_trip_count']
    avg_fare_val = analysis['avg_fare']
    passenger_cnt = analysis['avg_passenger_count']
    
    # Heuristic-based semantics
    if avg_trip_cnt > 500000 and avg_distance < 4:  # High volume, short trips
        semantic_label = "Downtown Core"
        description = "High-demand, high-frequency, short-distance trips (Manhattan CBD)"
    elif avg_distance > 15 and avg_fare_val > 35:  # Long trips, high fare
        semantic_label = "Airport/Long-Distance"
        description = "Long trips with high fares, likely airport or distant areas"
    elif avg_distance > 8 and avg_trip_cnt > 50000:  # Medium-long trips, decent volume
        semantic_label = "Hub/Connector"
        description = "Medium-distance trips connecting different zones"
    elif avg_trip_cnt < 20000:  # Low volume
        semantic_label = "Peripheral/Low-Demand"
        description = "Low-demand peripheral areas"
    elif passenger_cnt > 1.6:  # Group trips
        semantic_label = "Group/Shared"
        description = "Zones with higher average passenger count (group trips, hotels)"
    else:
        semantic_label = "Mixed/Residential"
        description = "Mixed characteristics, likely residential areas"
    
    cluster_semantics[int(cluster_id)] = {
        "label": semantic_label,
        "description": description
    }
    
    print(f"\nCluster {cluster_id}: {semantic_label}")
    print(f"  Description: {description}")
    print(f"  Zones: {analysis['num_zones']}")
    print(f"  Boroughs: {', '.join([f'{b} ({c})' for b, c in analysis['boroughs'].items()])}")
    print(f"  Avg trip count: {analysis['avg_trip_count']:,.0f}")
    print(f"  Avg distance: {analysis['avg_trip_distance']:.2f} miles")
    print(f"  Avg fare: ${analysis['avg_fare']:.2f}")
    print(f"  Avg passengers: {analysis['avg_passenger_count']:.2f}")
    print(f"  Top zones: {', '.join([z['Zone'] for z in analysis['top_zones']])}")

# Step 5: Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

try:
    # Save clustering model and results
    results = {
        'optimal_k': int(optimal_k),
        'zone_to_cluster': {int(k): int(v) for k, v in zone_to_cluster.items()},
        'silhouette_score': float(silhouette_scores[optimal_k]),
        'davies_bouldin_score': float(davies_bouldin_scores[optimal_k]),
        'all_silhouette_scores': {int(k): float(v) for k, v in silhouette_scores.items()},
        'all_davies_bouldin_scores': {int(k): float(v) for k, v in davies_bouldin_scores.items()},
        'cluster_centers': optimal_model.cluster_centers_.tolist(),
        'cluster_analysis': cluster_analysis,
        'cluster_semantics': cluster_semantics,
        'feature_names': clustering_cols,
        'feature_scaling': {
            'means': scaler.mean_.tolist(),
            'stds': scaler.scale_.tolist()
        }
    }
    
    # Save pickle file
    pkl_file = models_dir / "method2_clusters.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Saved model: {pkl_file}")
    
    # Save JSON report
    json_file = report_dir / "06_method2_clustering_report.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved report: {json_file}")
    
    # Save mobility features for analysis
    features_file = output_dir / "method2_mobility_features.parquet"
    mobility_df.to_parquet(features_file)
    print(f"✅ Saved features: {features_file}")
    
except Exception as e:
    print(f"❌ Error saving results: {e}")
    exit(1)

# Step 6: Create visualizations
print("\n[6/6] Creating visualizations...")

try:
    # Figure 1: Silhouette and Davies-Bouldin scores by K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    k_values = list(silhouette_scores.keys())
    sil_scores = list(silhouette_scores.values())
    db_scores = list(davies_bouldin_scores.values())
    
    # Silhouette plot
    ax1.plot(k_values, sil_scores, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    ax1.axhline(y=silhouette_scores[optimal_k], color='r', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title('Silhouette Score Analysis', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(k_values)
    
    # Davies-Bouldin plot (lower is better)
    ax2.plot(k_values, db_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    ax2.axhline(y=davies_bouldin_scores[optimal_k], color='r', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax2.set_ylabel('Davies-Bouldin Index', fontsize=11)
    ax2.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(k_values)
    
    fig.tight_layout()
    fig_file1 = figures_dir / "method2_clustering_metrics.png"
    plt.savefig(fig_file1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved figure: {fig_file1}")
    
    # Figure 2: Cluster characteristics heatmap
    cluster_profile = pd.DataFrame()
    for cluster_id in range(optimal_k):
        cluster_data = mobility_df[mobility_df['cluster'] == cluster_id]
        cluster_profile[f"Cluster {cluster_id}"] = [
            cluster_data['trip_count'].mean(),
            cluster_data['avg_trip_distance'].mean(),
            cluster_data['avg_fare'].mean(),
            cluster_data['avg_passenger_count'].mean()
        ]
    
    cluster_profile.index = ['Trip Count (log)', 'Avg Distance (mi)', 'Avg Fare ($)', 'Avg Passengers']
    
    # Normalize for heatmap
    cluster_profile_norm = (cluster_profile - cluster_profile.min(axis=1).values.reshape(-1, 1)) / \
                           (cluster_profile.max(axis=1).values.reshape(-1, 1) - cluster_profile.min(axis=1).values.reshape(-1, 1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_profile_norm, annot=cluster_profile, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Value'}, ax=ax)
    ax.set_title('Method 2: Cluster Characteristics', fontsize=13, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    
    fig.tight_layout()
    fig_file2 = figures_dir / "method2_cluster_heatmap.png"
    plt.savefig(fig_file2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved figure: {fig_file2}")
    
except Exception as e:
    print(f"⚠️ Warning: Error creating visualizations: {e}")

# Cleanup
conn.close()

print("\n" + "=" * 80)
print("✅ METHOD 2 CLUSTERING COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"  • Clustered {len(mobility_df)} zones by mobility patterns")
print(f"  • Optimal K: {optimal_k}")
print(f"  • Silhouette score: {silhouette_scores[optimal_k]:.4f}")
print(f"  • Davies-Bouldin score: {davies_bouldin_scores[optimal_k]:.4f}")
print(f"  • Features: {len(clustering_cols)} (trip volume, distance, fare, passengers, variability)")
print(f"\n📁 Outputs:")
print(f"  • Model: {pkl_file}")
print(f"  • Report: {json_file}")
print(f"  • Features: {features_file}")
print(f"  • Figures: {figures_dir / 'method2_*.png'}")
print(f"\n🔗 Next steps:")
print(f"  1. Method 3: OD-flow based clustering")
print(f"  2. Compare all methods (Baseline, Method 1, 2, 3)")
print(f"  3. Select top 2 methods for deep learning modeling")
print("\n")
