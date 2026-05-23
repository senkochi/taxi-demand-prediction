"""
Method 1: Demand-Based Clustering
Extract hourly demand patterns per zone and cluster zones with similar demand profiles
"""
import duckdb
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
db_path = Path("data/processed/taxi_features.duckdb")
output_dir = Path("data/processed/method1_features")
models_dir = Path("data/models")
report_dir = Path("logs")
figures_dir = Path("reports/figures")

# Create directories
output_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("METHOD 1: DEMAND-BASED CLUSTERING")
print("=" * 80)

# Connect to DuckDB
conn = duckdb.connect(str(db_path), read_only=True)

# Step 1: Extract hourly demand patterns
print("\n[1/5] Extracting hourly demand patterns...")
try:
    # For each zone and hour, sum the demand_count from baseline features
    hourly_demand_sql = """
    SELECT 
        zone_id,
        hour,
        SUM(demand_count) as hourly_demand
    FROM baseline_features
    GROUP BY zone_id, hour
    ORDER BY zone_id, hour
    """
    
    hourly_demand_df = conn.execute(hourly_demand_sql).df()
    print(f"✅ Extracted hourly demand for {hourly_demand_df['zone_id'].nunique()} zones")
    print(f"   Total hour-zone combinations: {len(hourly_demand_df):,}")
    
except Exception as e:
    print(f"❌ Error extracting hourly demand: {e}")
    exit(1)

# Step 2: Pivot to 24-hour profiles
print("\n[2/5] Creating 24-hour demand profiles...")
try:
    # Pivot to get 24 columns (one per hour)
    demand_profile = hourly_demand_df.pivot(
        index='zone_id', 
        columns='hour', 
        values='hourly_demand'
    ).fillna(0)
    
    # Handle missing hours (ensure all 0-23 are present)
    for hour in range(24):
        if hour not in demand_profile.columns:
            demand_profile[hour] = 0
    
    # Sort columns by hour
    demand_profile = demand_profile[[h for h in range(24)]]
    
    print(f"✅ Created demand profiles: {demand_profile.shape[0]} zones × 24 hours")
    print(f"\n   Sample demand profile (Zone 1):")
    if 1 in demand_profile.index:
        zone_1_demand = demand_profile.loc[1]
        for hour in [0, 6, 12, 18, 23]:
            print(f"     Hour {hour:2d}: {zone_1_demand[hour]:8.0f} trips")
    
except Exception as e:
    print(f"❌ Error creating demand profiles: {e}")
    exit(1)

# Step 3: Compute demand statistics
print("\n[3/5] Computing demand statistics...")
try:
    demand_features = pd.DataFrame({
        'zone_id': demand_profile.index,
        'demand_mean': demand_profile.mean(axis=1),
        'demand_std': demand_profile.std(axis=1),
        'demand_peak_hour': demand_profile.idxmax(axis=1),  # Hour with max demand
        'demand_peak_value': demand_profile.max(axis=1),
        'night_demand_ratio': demand_profile[[0, 1, 2, 3, 4, 5, 6]].sum(axis=1) / demand_profile.sum(axis=1),
        'morning_peak': demand_profile[[7, 8, 9, 10]].max(axis=1),  # Max in 7-10
        'midday_avg': demand_profile[[11, 12, 13, 14]].mean(axis=1),
        'evening_peak': demand_profile[[17, 18, 19, 20]].max(axis=1),  # Max in 17-20
        'night_early_ratio': demand_profile[[0, 1, 2, 3]].sum(axis=1) / demand_profile.sum(axis=1),
    })
    
    demand_features.reset_index(drop=True, inplace=True)
    
    print(f"✅ Computed features for {len(demand_features)} zones")
    print(f"\n   Feature statistics:")
    print(f"     demand_mean:         {demand_features['demand_mean'].mean():.1f} ± {demand_features['demand_mean'].std():.1f}")
    print(f"     demand_peak_hour:    {demand_features['demand_peak_hour'].mean():.1f} (hour of day)")
    print(f"     night_demand_ratio:  {demand_features['night_demand_ratio'].mean():.1%}")
    print(f"     morning_peak:        {demand_features['morning_peak'].mean():.1f}")
    print(f"     evening_peak:        {demand_features['evening_peak'].mean():.1f}")
    
except Exception as e:
    print(f"❌ Error computing statistics: {e}")
    exit(1)

# Step 4: Feature normalization
print("\n[4/5] Normalizing features for clustering...")
try:
    # Select features for clustering (exclude zone_id)
    clustering_features = demand_features[[
        'demand_mean', 'demand_std', 'demand_peak_hour', 'demand_peak_value',
        'night_demand_ratio', 'morning_peak', 'midday_avg', 'evening_peak', 'night_early_ratio'
    ]]
    
    # Handle any NaN values
    clustering_features = clustering_features.fillna(clustering_features.mean())
    
    # Standardize (zero mean, unit variance)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_features)
    
    print(f"✅ Normalized {scaled_features.shape[1]} features")
    print(f"   Shape: {scaled_features.shape[0]} zones × {scaled_features.shape[1]} features")
    
except Exception as e:
    print(f"❌ Error normalizing features: {e}")
    exit(1)

# Step 5: K-Means clustering with silhouette analysis
print("\n[5/5] Running K-Means with silhouette analysis...")
try:
    silhouette_scores = {}
    models = {}
    predictions_by_k = {}
    
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
        
        # Compute silhouette score
        sil_score = silhouette_score(scaled_features, labels)
        silhouette_scores[k] = sil_score
        models[k] = kmeans
        predictions_by_k[k] = labels
        
        print(f"Silhouette = {sil_score:.4f}")
    
    # Find optimal K
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    optimal_model = models[optimal_k]
    optimal_labels = predictions_by_k[optimal_k]
    
    print(f"\n✅ Silhouette analysis complete")
    print(f"   Optimal K: {optimal_k}")
    print(f"   Optimal silhouette score: {silhouette_scores[optimal_k]:.4f}")
    
    # Cluster distribution
    print(f"\n   Cluster distribution (K={optimal_k}):")
    unique, counts = np.unique(optimal_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = 100.0 * count / len(optimal_labels)
        print(f"     Cluster {cluster_id}: {count:3d} zones ({pct:5.1f}%)")
    
except Exception as e:
    print(f"❌ Error in K-Means clustering: {e}")
    exit(1)

# Step 6: Analyze clusters
print("\n" + "=" * 80)
print("CLUSTER ANALYSIS")
print("=" * 80)

demand_features['cluster'] = optimal_labels

# Zone to cluster mapping
zone_to_cluster = dict(zip(demand_features['zone_id'], demand_features['cluster']))

# Get zone info for cluster interpretation
conn2 = duckdb.connect(str(db_path), read_only=True)
zone_info_sql = """
SELECT DISTINCT zone_id, Borough, Zone, service_zone
FROM baseline_features
ORDER BY zone_id
"""
zone_info_df = conn2.execute(zone_info_sql).df()
conn2.close()

# Merge zone info with clusters
demand_features_enriched = demand_features.merge(
    zone_info_df,
    on='zone_id',
    how='left'
)

# Analyze each cluster
cluster_analysis = {}
for cluster_id in range(optimal_k):
    cluster_data = demand_features_enriched[demand_features_enriched['cluster'] == cluster_id]
    
    analysis = {
        'num_zones': len(cluster_data),
        'boroughs': cluster_data['Borough'].unique().tolist(),
        'avg_demand_mean': float(cluster_data['demand_mean'].mean()),
        'avg_morning_peak': float(cluster_data['morning_peak'].mean()),
        'avg_evening_peak': float(cluster_data['evening_peak'].mean()),
        'avg_night_ratio': float(cluster_data['night_demand_ratio'].mean()),
        'peak_hour': int(cluster_data['demand_peak_hour'].mode()[0]) if len(cluster_data) > 0 else -1,
        'sample_zones': cluster_data['Zone'].head(5).tolist()
    }
    
    cluster_analysis[int(cluster_id)] = analysis
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Zones: {analysis['num_zones']}")
    print(f"  Boroughs: {', '.join(analysis['boroughs'])}")
    print(f"  Avg demand: {analysis['avg_demand_mean']:.1f} trips/hour")
    print(f"  Morning peak: {analysis['avg_morning_peak']:.1f}")
    print(f"  Evening peak: {analysis['avg_evening_peak']:.1f}")
    print(f"  Night ratio: {analysis['avg_night_ratio']:.1%}")
    print(f"  Peak hour: {analysis['peak_hour']:02d}:00")
    print(f"  Sample zones: {', '.join(analysis['sample_zones'])}")

# Step 7: Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

try:
    # Save clustering model and results
    results = {
        'optimal_k': int(optimal_k),
        'zone_to_cluster': {int(k): int(v) for k, v in zone_to_cluster.items()},
        'silhouette_score': float(silhouette_scores[optimal_k]),
        'all_silhouette_scores': {int(k): float(v) for k, v in silhouette_scores.items()},
        'cluster_centers': optimal_model.cluster_centers_.tolist(),
        'cluster_analysis': cluster_analysis,
        'feature_names': [
            'demand_mean', 'demand_std', 'demand_peak_hour', 'demand_peak_value',
            'night_demand_ratio', 'morning_peak', 'midday_avg', 'evening_peak', 'night_early_ratio'
        ]
    }
    
    # Save pickle file
    pkl_file = models_dir / "method1_clusters.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Saved model: {pkl_file}")
    
    # Save JSON report
    json_file = report_dir / "05_method1_clustering_report.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved report: {json_file}")
    
    # Save demand features for analysis
    features_file = output_dir / "method1_demand_features.parquet"
    demand_features_enriched.to_parquet(features_file)
    print(f"✅ Saved features: {features_file}")
    
    # Save silhouette scores visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    k_values = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    
    ax.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    ax.axhline(y=silhouette_scores[optimal_k], color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Method 1: Silhouette Score by K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(k_values)
    
    fig_file = figures_dir / "method1_silhouette_scores.png"
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved figure: {fig_file}")
    
except Exception as e:
    print(f"❌ Error saving results: {e}")
    exit(1)

# Cleanup
conn.close()

print("\n" + "=" * 80)
print("✅ METHOD 1 CLUSTERING COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"  • Clustered {len(demand_features)} zones")
print(f"  • Optimal K: {optimal_k}")
print(f"  • Silhouette score: {silhouette_scores[optimal_k]:.4f}")
print(f"  • Features: 9 demand statistics (mean, std, peaks, ratios)")
print(f"\n📁 Outputs:")
print(f"  • Model: {pkl_file}")
print(f"  • Report: {json_file}")
print(f"  • Features: {features_file}")
print(f"  • Figure: {fig_file}")
print(f"\n🔗 Next steps:")
print(f"  1. Method 2: Mobility pattern clustering (trip distance, fare, passengers)")
print(f"  2. Method 3: OD-flow based clustering")
print(f"  3. Compare all methods with silhouette and Davies-Bouldin scores")
print("\n")
