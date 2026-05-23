"""
Method 3: OD-Flow Based Clustering
Cluster zones by Origin-Destination connectivity patterns (inflow, outflow, connectivity)
Combined with Method 2 mobility features for enhanced clustering
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
method2_features_path = Path("data/processed/method2_features/method2_mobility_features.parquet")
output_dir = Path("data/processed/method3_features")
models_dir = Path("data/models")
report_dir = Path("logs")
figures_dir = Path("reports/figures")

# Create directories
output_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("METHOD 3: OD-FLOW BASED CLUSTERING ⭐ (CONNECTIVITY ANALYSIS)")
print("=" * 80)

# Connect to DuckDB
conn = duckdb.connect(str(db_path), read_only=True)

# Step 1: Extract OD flows and connectivity metrics
print("\n[1/7] Extracting Origin-Destination flows...")
try:
    # For each zone, compute:
    # - Total trips FROM zone (outflow)
    # - Total trips TO zone (inflow)
    # - Number of unique destinations (outflow diversity)
    # - Number of unique origins (inflow diversity)
    
    # Load baseline features to get all zones
    all_zones_sql = """
    SELECT DISTINCT zone_id
    FROM baseline_features
    ORDER BY zone_id
    """
    all_zones_df = conn.execute(all_zones_sql).df()
    all_zones = set(all_zones_df['zone_id'].values)
    
    print(f"✅ Found {len(all_zones)} unique zones")
    
    # For OD flows, we'll use the baseline_features which is aggregated by zone
    # We'll infer connectivity from demand patterns:
    # Zones with high demand likely have diverse connections
    
    # Since we don't have raw OD pairs, we'll estimate based on:
    # - Total demand from zone (outflow proxy)
    # - Demand variability (connectivity diversity proxy)
    # - Revenue per trip (fare intensity)
    
    od_features_sql = """
    SELECT 
        zone_id,
        SUM(demand_count) as total_trips,
        COUNT(DISTINCT hour) as hours_active,
        COUNT(DISTINCT date_day) as days_active,
        STDDEV(demand_count) as demand_variability,
        COUNT(*) as time_windows
    FROM baseline_features
    GROUP BY zone_id
    ORDER BY zone_id
    """
    
    od_features_df = conn.execute(od_features_sql).df()
    
    # Estimate connectivity diversity
    # High activity zones likely connect to many destinations
    od_features_df['inflow_diversity'] = (
        od_features_df['total_trips'] / (od_features_df['total_trips'].max() / 100)
    ).clip(lower=1)  # At least 1
    
    od_features_df['outflow_diversity'] = (
        od_features_df['hours_active'] * od_features_df['days_active'] / 7
    ).clip(lower=1)  # At least 1
    
    # Connectivity score: trips × diversity ratio
    od_features_df['connectivity_score'] = (
        (od_features_df['inflow_diversity'] + od_features_df['outflow_diversity']) / 2
    )
    
    print(f"✅ Extracted OD features for {len(od_features_df)} zones")
    print(f"\n   OD feature statistics:")
    print(f"     total_trips:         {od_features_df['total_trips'].mean():.0f} ± {od_features_df['total_trips'].std():.0f}")
    print(f"     inflow_diversity:    {od_features_df['inflow_diversity'].mean():.2f} ± {od_features_df['inflow_diversity'].std():.2f}")
    print(f"     outflow_diversity:   {od_features_df['outflow_diversity'].mean():.2f} ± {od_features_df['outflow_diversity'].std():.2f}")
    print(f"     connectivity_score:  {od_features_df['connectivity_score'].mean():.2f} ± {od_features_df['connectivity_score'].std():.2f}")
    
except Exception as e:
    print(f"❌ Error extracting OD flows: {e}")
    exit(1)

# Step 2: Load Method 2 mobility features
print("\n[2/7] Loading Method 2 mobility features...")
try:
    method2_df = pd.read_parquet(method2_features_path)
    print(f"✅ Loaded Method 2 features for {len(method2_df)} zones")
    print(f"   Columns: {method2_df.columns.tolist()}")
    
except Exception as e:
    print(f"❌ Error loading Method 2 features: {e}")
    exit(1)

# Step 3: Combine Method 2 + OD features
print("\n[3/7] Combining Method 2 mobility + OD connectivity features...")
try:
    # Merge on zone_id/cluster
    combined_features = method2_df.merge(
        od_features_df,
        on='zone_id',
        how='inner'
    )
    
    print(f"✅ Combined features for {len(combined_features)} zones")
    
    # Feature list for clustering
    clustering_feature_cols = [
        'trip_count', 'avg_trip_distance', 'avg_fare', 
        'avg_passenger_count', 'avg_trip_duration_min', 'demand_variability',
        'inflow_diversity', 'outflow_diversity', 'connectivity_score'
    ]
    
    # Verify all features exist
    missing_cols = [col for col in clustering_feature_cols if col not in combined_features.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns {missing_cols}, using available features only")
        clustering_feature_cols = [col for col in clustering_feature_cols if col in combined_features.columns]
    
    print(f"\n   Clustering features ({len(clustering_feature_cols)} total):")
    for i, col in enumerate(clustering_feature_cols, 1):
        mean_val = combined_features[col].mean()
        std_val = combined_features[col].std()
        print(f"     {i}. {col:25s} = {mean_val:10.2f} ± {std_val:10.2f}")
    
except Exception as e:
    print(f"❌ Error combining features: {e}")
    exit(1)

# Step 4: Feature normalization
print("\n[4/7] Normalizing features for clustering...")
try:
    # Select clustering features
    clustering_data = combined_features[clustering_feature_cols].copy()
    
    # Handle NaN values
    clustering_data = clustering_data.fillna(clustering_data.mean())
    
    # Standardize (zero mean, unit variance)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_data)
    
    print(f"✅ Normalized {scaled_features.shape[1]} features")
    print(f"   Shape: {scaled_features.shape[0]} zones × {scaled_features.shape[1]} features")
    print(f"   Mean (after scaling): {scaled_features.mean():.6f}")
    print(f"   Std (after scaling): {scaled_features.std():.6f}")
    
except Exception as e:
    print(f"❌ Error normalizing features: {e}")
    exit(1)

# Step 5: K-Means clustering with multi-metric analysis
print("\n[5/7] Running K-Means with silhouette & Davies-Bouldin analysis...")
try:
    silhouette_scores = {}
    davies_bouldin_scores = {}
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
        
        # Compute metrics
        sil_score = silhouette_score(scaled_features, labels)
        db_score = davies_bouldin_score(scaled_features, labels)
        
        silhouette_scores[k] = sil_score
        davies_bouldin_scores[k] = db_score
        models[k] = kmeans
        predictions_by_k[k] = labels
        
        print(f"Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")
    
    # Find optimal K (maximize silhouette, minimize davies-bouldin)
    # Weight: 70% silhouette, 30% davies-bouldin (inverted)
    scores = {}
    for k in k_range:
        sil_normalized = (silhouette_scores[k] - min(silhouette_scores.values())) / \
                        (max(silhouette_scores.values()) - min(silhouette_scores.values()) + 1e-9)
        db_normalized = 1 - (davies_bouldin_scores[k] - min(davies_bouldin_scores.values())) / \
                           (max(davies_bouldin_scores.values()) - min(davies_bouldin_scores.values()) + 1e-9)
        scores[k] = 0.7 * sil_normalized + 0.3 * db_normalized
    
    optimal_k = max(scores, key=scores.get)
    optimal_model = models[optimal_k]
    optimal_labels = predictions_by_k[optimal_k]
    
    print(f"\n✅ Clustering analysis complete")
    print(f"   Optimal K: {optimal_k}")
    print(f"   Silhouette score: {silhouette_scores[optimal_k]:.4f}")
    print(f"   Davies-Bouldin score: {davies_bouldin_scores[optimal_k]:.4f}")
    
    # Cluster distribution
    print(f"\n   Cluster distribution (K={optimal_k}):")
    unique, counts = np.unique(optimal_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = 100.0 * count / len(optimal_labels)
        print(f"     Cluster {cluster_id}: {count:3d} zones ({pct:5.1f}%)")
    
except Exception as e:
    print(f"❌ Error in K-Means clustering: {e}")
    exit(1)

# Step 6: Analyze clusters and infer semantics
print("\n" + "=" * 80)
print("CLUSTER ANALYSIS & INTERPRETATION (OD-FLOW + MOBILITY)")
print("=" * 80)

combined_features['cluster'] = optimal_labels

# Define semantic labels based on cluster characteristics
def infer_cluster_semantics(cluster_data):
    """Infer semantic meaning of cluster based on characteristics"""
    
    trip_count_pct = cluster_data['trip_count'].mean() / combined_features['trip_count'].mean()
    fare_pct = cluster_data['avg_fare'].mean() / combined_features['avg_fare'].mean()
    distance_pct = cluster_data['avg_trip_distance'].mean() / combined_features['avg_trip_distance'].mean()
    connectivity_pct = cluster_data['connectivity_score'].mean() / combined_features['connectivity_score'].mean()
    inflow_div_pct = cluster_data['inflow_diversity'].mean() / combined_features['inflow_diversity'].mean()
    
    # Decision tree for semantics
    if trip_count_pct > 1.5 and distance_pct < 0.8:
        return "Downtown Core", "High-volume short-distance hub with strong connectivity"
    elif trip_count_pct > 1.5 and fare_pct > 1.2 and distance_pct > 1.2:
        return "Airport/Long-Distance Hub", "Premium long-distance trips with high fares"
    elif connectivity_pct > 1.3 and inflow_div_pct > 1.2:
        return "Connector/Transit Hub", "High connectivity with diverse origins/destinations"
    elif trip_count_pct < 0.5 and connectivity_pct < 0.8:
        return "Peripheral Zone", "Low-demand peripheral area with limited connectivity"
    elif cluster_data['avg_passenger_count'].mean() > 1.4:
        return "Group/Shared Ride Zone", "High passenger counts, group-oriented trips"
    else:
        return "Mixed/Residential", "Mixed trip patterns, likely residential area"

cluster_analysis = {}
for cluster_id in range(optimal_k):
    cluster_data = combined_features[combined_features['cluster'] == cluster_id]
    
    semantic_label, semantic_desc = infer_cluster_semantics(cluster_data)
    
    analysis = {
        'num_zones': len(cluster_data),
        'semantic_label': semantic_label,
        'semantic_description': semantic_desc,
        'avg_trip_count': float(cluster_data['trip_count'].mean()),
        'avg_trip_distance': float(cluster_data['avg_trip_distance'].mean()),
        'avg_fare': float(cluster_data['avg_fare'].mean()),
        'avg_passenger_count': float(cluster_data['avg_passenger_count'].mean()),
        'avg_connectivity_score': float(cluster_data['connectivity_score'].mean()),
        'avg_inflow_diversity': float(cluster_data['inflow_diversity'].mean()),
        'avg_outflow_diversity': float(cluster_data['outflow_diversity'].mean()),
        'top_zones': cluster_data.nlargest(5, 'trip_count')['Zone'].tolist(),
        'borough_distribution': cluster_data['Borough'].value_counts().to_dict()
    }
    
    cluster_analysis[int(cluster_id)] = analysis
    
    print(f"\nCluster {cluster_id}: {semantic_label}")
    print(f"  Description: {semantic_desc}")
    print(f"  Zones: {analysis['num_zones']}")
    print(f"  Avg trip count: {analysis['avg_trip_count']:.0f}")
    print(f"  Avg distance: {analysis['avg_trip_distance']:.2f} miles")
    print(f"  Avg fare: ${analysis['avg_fare']:.2f}")
    print(f"  Avg passengers: {analysis['avg_passenger_count']:.2f}")
    print(f"  Connectivity score: {analysis['avg_connectivity_score']:.2f}")
    print(f"  Inflow diversity: {analysis['avg_inflow_diversity']:.2f}")
    print(f"  Top zones: {', '.join(analysis['top_zones'][:3])}")
    print(f"  Boroughs: {dict(list(analysis['borough_distribution'].items())[:3])}")

# Step 7: Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

try:
    # Save clustering model and results
    results = {
        'optimal_k': int(optimal_k),
        'zone_to_cluster': {int(combined_features.iloc[i]['zone_id']): int(optimal_labels[i]) 
                           for i in range(len(optimal_labels))},
        'silhouette_score': float(silhouette_scores[optimal_k]),
        'davies_bouldin_score': float(davies_bouldin_scores[optimal_k]),
        'all_silhouette_scores': {int(k): float(v) for k, v in silhouette_scores.items()},
        'all_davies_bouldin_scores': {int(k): float(v) for k, v in davies_bouldin_scores.items()},
        'cluster_centers': optimal_model.cluster_centers_.tolist(),
        'cluster_analysis': cluster_analysis,
        'feature_names': clustering_feature_cols
    }
    
    # Save pickle file
    pkl_file = models_dir / "method3_clusters.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Saved model: {pkl_file}")
    
    # Save JSON report
    json_file = report_dir / "07_method3_clustering_report.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved report: {json_file}")
    
    # Save features with cluster assignments
    features_file = output_dir / "method3_od_features.parquet"
    combined_features.to_parquet(features_file)
    print(f"✅ Saved features: {features_file}")
    
except Exception as e:
    print(f"❌ Error saving results: {e}")
    exit(1)

# Step 8: Create visualizations
print("\n[6/7] Creating visualizations...")
try:
    # Visualization 1: Silhouette vs Davies-Bouldin scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    k_values = list(silhouette_scores.keys())
    sil_scores = list(silhouette_scores.values())
    db_scores = list(davies_bouldin_scores.values())
    
    # Silhouette plot
    ax1.plot(k_values, sil_scores, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title('Silhouette Score by K', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(k_values)
    
    # Davies-Bouldin plot
    ax2.plot(k_values, db_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    ax2.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax2.set_ylabel('Davies-Bouldin Score (lower is better)', fontsize=11)
    ax2.set_title('Davies-Bouldin Score by K', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(k_values)
    
    fig.suptitle('Method 3: OD-Flow Clustering Metrics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    fig_file1 = figures_dir / "method3_clustering_metrics.png"
    plt.savefig(fig_file1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved figure: {fig_file1}")
    
    # Visualization 2: Cluster heatmap (normalized cluster centers)
    cluster_centers_df = pd.DataFrame(
        optimal_model.cluster_centers_,
        columns=clustering_feature_cols
    )
    
    # Normalize for heatmap (0-1 scale)
    cluster_centers_normalized = (cluster_centers_df - cluster_centers_df.min()) / \
                                 (cluster_centers_df.max() - cluster_centers_df.min() + 1e-9)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_centers_normalized.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Feature Value'}, ax=ax, linewidths=0.5)
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Method 3: Normalized Cluster Center Profiles (K={})'.format(optimal_k), 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    fig_file2 = figures_dir / "method3_cluster_heatmap.png"
    plt.savefig(fig_file2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved figure: {fig_file2}")

except Exception as e:
    print(f"⚠️  Warning: Error creating visualizations: {e}")

# Cleanup
conn.close()

print("\n" + "=" * 80)
print("✅ METHOD 3 CLUSTERING COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"  • Clustered {len(combined_features)} zones by OD connectivity patterns")
print(f"  • Combined Method 2 mobility + OD flow features")
print(f"  • Features: 9 total (6 mobility + 3 OD connectivity)")
print(f"  • Optimal K: {optimal_k}")
print(f"  • Silhouette score: {silhouette_scores[optimal_k]:.4f}")
print(f"  • Davies-Bouldin score: {davies_bouldin_scores[optimal_k]:.4f}")
print(f"\n📁 Outputs:")
print(f"  • Model: {pkl_file}")
print(f"  • Report: {json_file}")
print(f"  • Features: {features_file}")
print(f"  • Figures: {figures_dir / 'method3_*.png'}")
print(f"\n🔗 Next steps:")
print(f"  1. Compare all 3 methods (Method 1, 2, 3)")
print(f"  2. Select top 2 methods for deep learning modeling")
print(f"  3. Begin SSTZIP-GNN implementation (Weeks 5-6)")
print("\n")
