"""
Inference utilities for test set evaluation.

This module handles test set evaluation including cluster assignment
and performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .metrics import compute_precision_at_k, compute_all_ranking_metrics


def evaluate_test_set(
    y_test: np.ndarray,
    predictions_test: np.ndarray,
    clusters_test: Optional[np.ndarray] = None,
    cluster_success_rates: Optional[List[Tuple[int, float]]] = None,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive test set evaluation.
    
    Args:
        y_test: Test set ground truth labels
        predictions_test: Test set predictions
        clusters_test: Optional cluster assignments for test set
        cluster_success_rates: Optional list of (cluster_id, success_rate) tuples
        verbose: Whether to print results
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}
    
    # Ranking metrics
    ranking_metrics = compute_all_ranking_metrics(y_test, predictions_test)
    results['ranking'] = ranking_metrics
    
    if verbose:
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        print(f"\nBase rate: {ranking_metrics['base_rate']:.1%}")
        print(f"Total successful: {y_test.sum()}/{len(y_test)}")
        print("\nRanking Performance:")
        for k in [10, 20, 50, 100, 200]:
            key = f'p@{k}'
            if key in ranking_metrics:
                lift = ranking_metrics[key] / ranking_metrics['base_rate']
                print(f"  P@{k:3d}: {ranking_metrics[key]:.1%} (lift: {lift:.2f}x)")
    
    # Cluster-based metrics if available
    if clusters_test is not None and cluster_success_rates is not None:
        cluster_metrics = _evaluate_cluster_strategies(
            y_test, clusters_test, cluster_success_rates, verbose
        )
        results['cluster'] = cluster_metrics
    
    return results


def _evaluate_cluster_strategies(
    y_test: np.ndarray,
    clusters_test: np.ndarray,
    cluster_success_rates: List[Tuple[int, float]],
    verbose: bool = True
) -> Dict:
    """Evaluate cluster-based selection strategies."""
    n_clusters = len(cluster_success_rates)
    n_successful = y_test.sum()
    results = {}
    
    if verbose:
        print("\n" + "="*80)
        print("CLUSTER-BASED STRATEGIES")
        print("="*80)
    
    # Strategy 1: TOP 1 cluster
    high_success_cluster = cluster_success_rates[0][0]
    in_top1 = (clusters_test == high_success_cluster)
    
    if in_top1.sum() > 0:
        precision_top1 = y_test[in_top1].mean()
        coverage_top1 = in_top1.sum() / len(y_test)
        captured_top1 = (y_test[in_top1] == 1).sum()
        
        results['top1'] = {
            'cluster': int(high_success_cluster),
            'selected': int(in_top1.sum()),
            'precision': float(precision_top1),
            'coverage': float(coverage_top1),
            'lift': float(precision_top1 / y_test.mean()),
            'captured': int(captured_top1),
            'captured_pct': float(captured_top1 / n_successful) if n_successful > 0 else 0
        }
        
        if verbose:
            print(f"\nStrategy: TOP 1 cluster (Cluster {high_success_cluster}):")
            print(f"  Founders selected: {in_top1.sum()}")
            print(f"  Precision: {precision_top1:.1%}")
            print(f"  Coverage: {coverage_top1:.1%}")
            print(f"  Lift: {precision_top1 / y_test.mean():.2f}x")
            print(f"  Captured: {captured_top1}/{n_successful} ({captured_top1/n_successful*100:.1f}%)")
    
    # Strategy 2: TOP 2 clusters
    if n_clusters >= 2:
        top2_clusters = [cluster_success_rates[i][0] for i in range(2)]
        in_top2 = np.isin(clusters_test, top2_clusters)
        
        if in_top2.sum() > 0:
            precision_top2 = y_test[in_top2].mean()
            coverage_top2 = in_top2.sum() / len(y_test)
            captured_top2 = (y_test[in_top2] == 1).sum()
            
            results['top2'] = {
                'clusters': top2_clusters,
                'selected': int(in_top2.sum()),
                'precision': float(precision_top2),
                'coverage': float(coverage_top2),
                'lift': float(precision_top2 / y_test.mean()),
                'captured': int(captured_top2),
                'captured_pct': float(captured_top2 / n_successful) if n_successful > 0 else 0
            }
            
            if verbose:
                print(f"\nStrategy: TOP 2 clusters (Clusters {top2_clusters}):")
                print(f"  Founders selected: {in_top2.sum()}")
                print(f"  Precision: {precision_top2:.1%}")
                print(f"  Lift: {precision_top2 / y_test.mean():.2f}x")
                print(f"  Captured: {captured_top2}/{n_successful} ({captured_top2/n_successful*100:.1f}%)")
    
    # Strategy 3: TOP 3 clusters
    if n_clusters >= 3:
        top3_clusters = [cluster_success_rates[i][0] for i in range(3)]
        in_top3 = np.isin(clusters_test, top3_clusters)
        
        if in_top3.sum() > 0:
            precision_top3 = y_test[in_top3].mean()
            coverage_top3 = in_top3.sum() / len(y_test)
            captured_top3 = (y_test[in_top3] == 1).sum()
            
            results['top3'] = {
                'clusters': top3_clusters,
                'selected': int(in_top3.sum()),
                'precision': float(precision_top3),
                'coverage': float(coverage_top3),
                'lift': float(precision_top3 / y_test.mean()),
                'captured': int(captured_top3),
                'captured_pct': float(captured_top3 / n_successful) if n_successful > 0 else 0
            }
            
            if verbose:
                print(f"\nStrategy: TOP 3 clusters (Clusters {top3_clusters}):")
                print(f"  Founders selected: {in_top3.sum()}")
                print(f"  Precision: {precision_top3:.1%}")
                print(f"  Lift: {precision_top3 / y_test.mean():.2f}x")
                print(f"  Captured: {captured_top3}/{n_successful} ({captured_top3/n_successful*100:.1f}%)")
    
    return results


def assign_test_to_clusters(
    embeddings_test: np.ndarray,
    baseline_test: np.ndarray,
    scaler_embeddings: StandardScaler,
    scaler_baseline: StandardScaler,
    kmeans: KMeans,
    verbose: bool = True
) -> np.ndarray:
    """
    Assign test set founders to trained clusters.
    
    Args:
        embeddings_test: Test set GNN embeddings
        baseline_test: Test set baseline features
        scaler_embeddings: Fitted scaler for embeddings
        scaler_baseline: Fitted scaler for baseline features
        kmeans: Fitted KMeans model
        verbose: Whether to print progress
        
    Returns:
        Array of cluster assignments for test set
    """
    if verbose:
        print("\n[Inference] Assigning test set to clusters...")
    
    # Scale features using training scalers
    embeddings_test_scaled = scaler_embeddings.transform(embeddings_test)
    baseline_test_scaled = scaler_baseline.transform(baseline_test)
    
    # Combine features
    combined_test = np.concatenate([embeddings_test_scaled, baseline_test_scaled], axis=1)
    
    if verbose:
        print(f"✓ Combined test features: {combined_test.shape}")
    
    # Assign to clusters
    clusters_test = kmeans.predict(combined_test)
    
    if verbose:
        print(f"✓ Assigned {len(clusters_test)} test founders to {kmeans.n_clusters} clusters")
    
    return clusters_test


def analyze_cluster_performance(
    y_train: np.ndarray,
    clusters_train: np.ndarray,
    y_test: np.ndarray,
    clusters_test: np.ndarray,
    n_clusters: int,
    verbose: bool = True
) -> Tuple[List[Tuple[int, float]], Dict]:
    """
    Analyze cluster performance comparing train and test sets.
    
    Args:
        y_train: Training set labels
        clusters_train: Training set cluster assignments
        y_test: Test set labels
        clusters_test: Test set cluster assignments
        n_clusters: Number of clusters
        verbose: Whether to print results
        
    Returns:
        Tuple of (sorted cluster success rates, detailed stats)
    """
    cluster_success_rates = []
    cluster_stats = {}
    
    if verbose:
        print("\nCluster Distribution Comparison:")
        print(f"{'Cluster':<10} {'Train Size':<15} {'Test Size':<15} {'Train Success':<18} {'Test Success':<18}")
        print("-"*85)
    
    for c in range(n_clusters):
        train_mask_c = (clusters_train == c)
        test_mask_c = (clusters_test == c)
        
        train_size = train_mask_c.sum()
        test_size = test_mask_c.sum()
        
        train_success = y_train[train_mask_c].mean() if train_size > 0 else 0
        test_success = y_test[test_mask_c].mean() if test_size > 0 else 0
        
        cluster_success_rates.append((c, train_success))
        cluster_stats[c] = {
            'train_size': int(train_size),
            'test_size': int(test_size),
            'train_success_rate': float(train_success),
            'test_success_rate': float(test_success)
        }
        
        if verbose:
            print(f"{c:<10} {train_size:<15} {test_size:<15} {train_success:<18.1%} {test_success:<18.1%}")
    
    # Sort by success rate
    cluster_success_rates.sort(key=lambda x: x[1], reverse=True)
    
    if verbose:
        print(f"\nClusters ranked by training success rate:")
        for i, (c, rate) in enumerate(cluster_success_rates):
            if i == 0:
                status = "🟢 HIGH"
            elif i < n_clusters // 2:
                status = "🟡 MED-HIGH"
            elif i < (n_clusters * 2) // 3:
                status = "🟠 MED"
            else:
                status = "🔴 LOW"
            print(f"  {status}: Cluster {c} ({rate:.1%})")
    
    return cluster_success_rates, cluster_stats


def where_do_successful_founders_go(
    y_test: np.ndarray,
    clusters_test: np.ndarray,
    cluster_success_rates: List[Tuple[int, float]],
    verbose: bool = True
) -> Dict:
    """
    Analyze distribution of successful test founders across clusters.
    
    Args:
        y_test: Test set labels
        clusters_test: Test set cluster assignments
        cluster_success_rates: Sorted list of (cluster_id, success_rate)
        verbose: Whether to print results
        
    Returns:
        Dictionary with distribution statistics
    """
    n_clusters = len(cluster_success_rates)
    successful_test = (y_test == 1)
    n_successful = successful_test.sum()
    
    distribution = {}
    
    if verbose:
        print("\n" + "="*80)
        print("WHERE DO SUCCESSFUL TEST FOUNDERS END UP?")
        print("="*80)
        print(f"\nTotal successful test founders: {n_successful}")
        print(f"\nDistribution across clusters:")
    
    for c in range(n_clusters):
        mask = (clusters_test == c) & successful_test
        count = mask.sum()
        pct = count / n_successful * 100 if n_successful > 0 else 0
        
        # Find cluster rank
        cluster_rank = next(i for i, (cluster_id, _) in enumerate(cluster_success_rates) if cluster_id == c)
        
        if cluster_rank == 0:
            status = "🟢 HIGH SUCCESS CLUSTER"
        elif cluster_rank < n_clusters // 2:
            status = "🟡 MED-HIGH SUCCESS CLUSTER"
        elif cluster_rank < (n_clusters * 2) // 3:
            status = "🟠 MED SUCCESS CLUSTER"
        else:
            status = "🔴 LOW SUCCESS CLUSTER"
        
        distribution[c] = {
            'count': int(count),
            'percentage': float(pct),
            'rank': cluster_rank,
            'status': status
        }
        
        if verbose:
            print(f"  Cluster {c} ({status}): {count:3d} successful founders ({pct:5.1f}%)")
    
    return distribution
