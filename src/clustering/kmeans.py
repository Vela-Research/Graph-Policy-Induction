"""
K-means clustering for founder archetypes.

This module provides clustering functionality to identify
distinct founder archetypes based on GNN embeddings and
baseline features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_clusters(
    features: np.ndarray,
    k_range: range = range(2, 11),
    random_seed: int = 42,
    n_init: int = 10,
    verbose: bool = True
) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        features: Combined feature matrix
        k_range: Range of K values to try
        random_seed: Random seed for reproducibility
        n_init: Number of initializations per K
        verbose: Whether to print progress
        
    Returns:
        Tuple of (optimal K, list of silhouette scores)
    """
    if verbose:
        print("\n[Clustering] Finding optimal number of clusters...")
    
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=n_init)
        cluster_labels = kmeans.fit_predict(features)
        score = silhouette_score(features, cluster_labels)
        silhouette_scores.append(score)
        
        if verbose:
            print(f"  k={k}: silhouette score = {score:.4f}")
    
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    
    if verbose:
        print(f"\n✓ Optimal number of clusters: {optimal_k}")
        print(f"  Best silhouette score: {best_score:.4f}")
    
    return optimal_k, silhouette_scores


def perform_clustering(
    embeddings: np.ndarray,
    baseline_features: pd.DataFrame,
    n_clusters: int,
    random_seed: int = 42,
    n_init: int = 20,
    verbose: bool = True
) -> Tuple[np.ndarray, KMeans, StandardScaler, StandardScaler, np.ndarray]:
    """
    Perform K-means clustering on combined features.
    
    Args:
        embeddings: GNN embeddings
        baseline_features: Baseline feature DataFrame
        n_clusters: Number of clusters
        random_seed: Random seed
        n_init: Number of KMeans initializations
        verbose: Whether to print progress
        
    Returns:
        Tuple of (cluster assignments, fitted KMeans, embedding scaler, 
                  baseline scaler, combined features)
    """
    if verbose:
        print(f"\n[Clustering] Clustering founders into {n_clusters} archetypes...")
    
    # Standardize features
    scaler_embeddings = StandardScaler()
    scaler_baseline = StandardScaler()
    
    embeddings_scaled = scaler_embeddings.fit_transform(embeddings)
    baseline_scaled = scaler_baseline.fit_transform(baseline_features.fillna(0))
    
    if verbose:
        print(f"✓ Standardized embeddings: {embeddings_scaled.shape}")
        print(f"✓ Standardized baseline: {baseline_scaled.shape}")
    
    # Combine features
    combined_features = np.concatenate([embeddings_scaled, baseline_scaled], axis=1)
    
    if verbose:
        print(f"✓ Combined features: {combined_features.shape}")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=n_init)
    clusters = kmeans.fit_predict(combined_features)
    
    if verbose:
        print(f"✓ Clustered {len(clusters)} founders into {n_clusters} archetypes")
        for c in range(n_clusters):
            n_in_cluster = (clusters == c).sum()
            pct = n_in_cluster / len(clusters) * 100
            print(f"  Cluster {c}: {n_in_cluster:4d} founders ({pct:5.1f}%)")
    
    return clusters, kmeans, scaler_embeddings, scaler_baseline, combined_features


def analyze_clusters(
    clusters: np.ndarray,
    y: np.ndarray,
    gnn_predictions: np.ndarray,
    baseline_features: pd.DataFrame,
    n_clusters: int,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Analyze characteristics of each cluster.
    
    Args:
        clusters: Cluster assignments
        y: Success labels
        gnn_predictions: GNN prediction scores
        baseline_features: Baseline feature DataFrame
        n_clusters: Number of clusters
        verbose: Whether to print analysis
        
    Returns:
        Tuple of (cluster descriptions dict, cluster stats dict)
    """
    cluster_descriptions = {}
    cluster_stats = {}
    
    n_founders = len(clusters)
    
    for c in range(n_clusters):
        cluster_mask = (clusters == c)
        n_in_cluster = cluster_mask.sum()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"CLUSTER {c}")
            print(f"{'='*80}")
        
        # Basic stats
        success_rate = y[cluster_mask].mean()
        avg_gnn_score = gnn_predictions[cluster_mask].mean()
        
        if verbose:
            print(f"Size: {n_in_cluster} founders ({n_in_cluster/n_founders*100:.1f}%)")
            print(f"Success Rate: {success_rate:.1%}")
            print(f"Avg GNN Score: {avg_gnn_score:.1%}")
        
        # Feature analysis
        cluster_baseline = baseline_features[cluster_mask].mean()
        overall_mean = baseline_features.mean()
        
        deviations = abs(cluster_baseline - overall_mean)
        top_features = deviations.nlargest(15)
        
        if verbose:
            print(f"\nDistinguishing Features:")
            for feat in top_features.index:
                cluster_val = cluster_baseline[feat]
                overall_val = overall_mean[feat]
                ratio = cluster_val / (overall_val + 1e-10)
                direction = "↑" if cluster_val > overall_val else "↓"
                print(f"  {direction} {feat:40s}: {cluster_val:6.2f} (avg: {overall_val:6.2f}, {ratio:.2f}x)")
        
        # Generate description
        description = _generate_cluster_description(
            success_rate, avg_gnn_score, gnn_predictions.mean(),
            cluster_baseline, overall_mean, top_features
        )
        
        cluster_descriptions[c] = description
        cluster_stats[c] = {
            'size': int(n_in_cluster),
            'success_rate': float(success_rate),
            'avg_gnn_score': float(avg_gnn_score),
            'description': description
        }
        
        if verbose:
            print(f"\n✓ Cluster Label: {description}")
    
    return cluster_descriptions, cluster_stats


def _generate_cluster_description(
    success_rate: float,
    avg_gnn_score: float,
    overall_gnn_mean: float,
    cluster_baseline: pd.Series,
    overall_mean: pd.Series,
    top_features: pd.Series
) -> str:
    """Generate a descriptive label for a cluster."""
    description_parts = []
    
    # Success rate label
    if success_rate > 0.15:
        description_parts.append("High Success")
    elif success_rate > 0.08:
        description_parts.append("Medium Success")
    else:
        description_parts.append("Low Success")
    
    # GNN score label
    if avg_gnn_score > overall_gnn_mean * 1.2:
        description_parts.append("Strong Network")
    elif avg_gnn_score < overall_gnn_mean * 0.8:
        description_parts.append("Weak Network")
    
    # Feature-based labels
    for feat in top_features.index[:3]:
        cluster_val = cluster_baseline[feat]
        overall_val = overall_mean[feat]
        
        if cluster_val > overall_val * 1.3:
            if 'edu' in feat.lower() and 'top' in feat.lower():
                description_parts.append("Elite Education")
            elif 'senior' in feat.lower() or 'leadership' in feat.lower():
                description_parts.append("Senior Leadership")
            elif 'experience' in feat.lower() or 'years' in feat.lower():
                description_parts.append("Experienced")
            elif 'tech' in feat.lower():
                description_parts.append("Tech Background")
        elif cluster_val < overall_val * 0.7:
            if 'experience' in feat.lower() or 'years' in feat.lower():
                description_parts.append("Early Career")
    
    # Remove duplicates and join
    description_parts = list(dict.fromkeys(description_parts))
    return " | ".join(description_parts) if description_parts else "General"


def get_cluster_descriptions(
    clusters: np.ndarray,
    cluster_descriptions: Dict[int, str]
) -> List[str]:
    """
    Get description for each founder based on cluster assignment.
    
    Args:
        clusters: Cluster assignments
        cluster_descriptions: Dictionary mapping cluster ID to description
        
    Returns:
        List of descriptions for each founder
    """
    return [cluster_descriptions[c] for c in clusters]


def save_cluster_results(
    clusters: np.ndarray,
    cluster_descriptions: Dict[int, str],
    gnn_predictions: np.ndarray,
    y: np.ndarray,
    output_path: str,
    verbose: bool = True
):
    """
    Save cluster assignments and metadata.
    
    Args:
        clusters: Cluster assignments
        cluster_descriptions: Cluster description dictionary
        gnn_predictions: GNN predictions
        y: Success labels
        output_path: Output file path
        verbose: Whether to print progress
    """
    import json
    from pathlib import Path
    
    n_founders = len(clusters)
    
    # Save assignments CSV
    cluster_df = pd.DataFrame({
        'founder_idx': range(n_founders),
        'cluster': clusters,
        'cluster_description': [cluster_descriptions[c] for c in clusters],
        'gnn_prediction': gnn_predictions,
        'success': y
    })
    
    csv_path = Path(output_path) / 'founder_cluster_assignments.csv'
    cluster_df.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"✓ Saved {csv_path}")
    
    # Save cluster statistics
    cluster_stats = {}
    for c in range(len(cluster_descriptions)):
        mask = (clusters == c)
        cluster_stats[str(c)] = {
            'size': int(mask.sum()),
            'success_rate': float(y[mask].mean()),
            'avg_prediction': float(gnn_predictions[mask].mean()),
            'description': cluster_descriptions[c]
        }
    
    json_path = Path(output_path) / 'cluster_statistics.json'
    with open(json_path, 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    
    if verbose:
        print(f"✓ Saved {json_path}")
