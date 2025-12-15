"""
Visualization functions for founder success prediction.

This module provides plotting utilities for:
- Cluster analysis
- Training metrics
- Test set evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_cluster_optimization(
    k_range: range,
    silhouette_scores: List[float],
    optimal_k: int,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot silhouette scores for different numbers of clusters.
    
    Args:
        k_range: Range of K values tested
        silhouette_scores: Silhouette score for each K
        optimal_k: Optimal number of clusters
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(list(k_range), silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color='red', linestyle='--', 
               label=f'Optimal k={optimal_k}')
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Optimal Number of Clusters', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_cluster_visualization(
    combined_features: np.ndarray,
    clusters: np.ndarray,
    y: np.ndarray,
    cluster_descriptions: Dict[int, str],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create 2D visualization of founder clusters using PCA.
    
    Args:
        combined_features: Combined feature matrix
        clusters: Cluster assignments
        y: Success labels
        cluster_descriptions: Cluster description dictionary
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    n_clusters = len(cluster_descriptions)
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_features)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Clusters
    ax = axes[0]
    for c in range(n_clusters):
        cluster_mask = (clusters == c)
        label = f"C{c}: {cluster_descriptions[c][:30]}"
        ax.scatter(combined_pca[cluster_mask, 0], combined_pca[cluster_mask, 1],
                   label=label, alpha=0.6, s=30)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax.set_title('Founder Archetypes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Color by success
    ax = axes[1]
    scatter = ax.scatter(combined_pca[:, 0], combined_pca[:, 1],
                         c=y, cmap='RdYlGn', alpha=0.6, s=30,
                         vmin=0, vmax=1)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax.set_title('Colored by Success', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Success')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_test_evaluation(
    combined_train: np.ndarray,
    combined_test: np.ndarray,
    clusters_train: np.ndarray,
    clusters_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cluster_success_rates: List[Tuple[int, float]],
    n_clusters: int,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create visualization of test set cluster assignments.
    
    Args:
        combined_train: Combined features for training set
        combined_test: Combined features for test set
        clusters_train: Training set cluster assignments
        clusters_test: Test set cluster assignments
        y_train: Training set labels
        y_test: Test set labels
        cluster_success_rates: Sorted list of (cluster_id, success_rate)
        n_clusters: Number of clusters
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # PCA
    pca = PCA(n_components=2)
    combined_all = np.concatenate([combined_train, combined_test], axis=0)
    combined_pca = pca.fit_transform(combined_all)
    
    n_train = len(combined_train)
    train_pca = combined_pca[:n_train]
    test_pca = combined_pca[n_train:]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Train clusters
    ax = axes[0]
    for c in range(n_clusters):
        mask = (clusters_train == c)
        success_rate = y_train[mask].mean()
        cluster_rank = next(i for i, (cluster_id, _) in enumerate(cluster_success_rates) if cluster_id == c)
        
        if cluster_rank == 0:
            emoji = "🟢"
        elif cluster_rank < n_clusters // 2:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        label = f"C{c} ({success_rate:.0%}) {emoji}"
        ax.scatter(train_pca[mask, 0], train_pca[mask, 1], label=label, alpha=0.6, s=30)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('TRAIN+VAL (Cluster Training)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test assignments
    ax = axes[1]
    for c in range(n_clusters):
        mask = (clusters_test == c)
        ax.scatter(test_pca[mask, 0], test_pca[mask, 1], label=f"C{c}", alpha=0.6, s=30)
    
    # Highlight successful founders
    successful_mask = (y_test == 1)
    ax.scatter(test_pca[successful_mask, 0], test_pca[successful_mask, 1],
               marker='*', s=200, c='red', edgecolors='black', linewidths=1,
               label='Actual Success', zorder=10)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('TEST SET (Cluster Assignments)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_training_history(
    history: Dict,
    target_k: int = 100,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training loss and validation metrics.
    
    Args:
        history: Training history dictionary
        target_k: K value used in training
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training loss
    ax = axes[0]
    ax.plot(history['loss'], 'b-', alpha=0.7, linewidth=1)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation P@K
    ax = axes[1]
    pk_key = f'val_p@{target_k}'
    if pk_key in history:
        eval_epochs = range(0, len(history['loss']), len(history['loss']) // len(history[pk_key]) if len(history[pk_key]) > 1 else 1)
        ax.plot(list(eval_epochs)[:len(history[pk_key])], history[pk_key], 'go-', 
                linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(f'P@{target_k}', fontsize=11)
    ax.set_title(f'Validation P@{target_k}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_precision_at_k(
    results: Dict,
    base_rate: float,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot precision at various K values.
    
    Args:
        results: Dictionary with P@K results
        base_rate: Base success rate
        model_name: Name of the model
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    k_values = []
    precisions = []
    
    for k in [10, 20, 50, 100, 200, 500]:
        key = f'p@{k}'
        if key in results:
            k_values.append(k)
            if isinstance(results[key], dict):
                precisions.append(results[key]['mean'])
            else:
                precisions.append(results[key])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot P@K
    ax.plot(k_values, precisions, 'bo-', linewidth=2, markersize=10, label=model_name)
    
    # Plot base rate
    ax.axhline(y=base_rate, color='red', linestyle='--', linewidth=2, 
               label=f'Base Rate ({base_rate:.1%})')
    
    # Add lift annotations
    for k, p in zip(k_values, precisions):
        lift = p / base_rate
        ax.annotate(f'{lift:.1f}x', (k, p), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    ax.set_xlabel('K (Top-K Predictions)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{model_name}: Precision at K', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_cluster_success_comparison(
    cluster_stats: Dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot success rate comparison across clusters.
    
    Args:
        cluster_stats: Dictionary with cluster statistics
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    clusters = list(cluster_stats.keys())
    success_rates = [cluster_stats[c]['success_rate'] for c in clusters]
    sizes = [cluster_stats[c]['size'] for c in clusters]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(clusters, success_rates, color='steelblue', alpha=0.8)
    
    # Add size labels
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.annotate(f'n={size}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate by Cluster', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_path}")
    
    if show:
        plt.show()
    
    return fig
