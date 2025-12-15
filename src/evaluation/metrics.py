"""
Evaluation metrics for founder success prediction.

This module provides metrics aligned with Vela's evaluation framework,
prioritizing precision over recall (P@K, F0.5, Lift).
"""

import numpy as np
from typing import Dict, Optional


def compute_precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """
    Compute precision at top-K predictions.
    
    This is the primary metric for ranking-based evaluation.
    No threshold needed - simply rank by probability and take top K.
    
    Args:
        y_true: Binary ground truth labels
        y_proba: Predicted probabilities
        k: Number of top predictions to consider
        
    Returns:
        Precision among top K predictions
    """
    if k > len(y_true):
        k = len(y_true)
    top_k_idx = np.argsort(y_proba)[-k:]
    return y_true[top_k_idx].mean()


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Compute F-beta score.
    
    F0.5 weights precision MORE than recall (Vela's preferred metric).
    From GPTree paper: "we prioritize precision over recall"
    
    Args:
        precision: Precision value
        recall: Recall value
        beta: Beta parameter (0.5 = precision-focused)
        
    Returns:
        F-beta score
    """
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_precision_recall_f05(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute precision, recall, and F0.5 score.
    
    Args:
        y_true: Binary ground truth labels
        y_pred: Binary predictions
        
    Returns:
        Dictionary with precision, recall, F0.5, and counts
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = compute_f_beta(precision, recall, beta=0.5)
    
    return {
        'precision': precision, 
        'recall': recall, 
        'f05': f05,
        'tp': int(tp), 
        'fp': int(fp), 
        'fn': int(fn)
    }


def compute_all_ranking_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict:
    """
    Compute all ranking metrics (P@K for various K values).
    
    Args:
        y_true: Binary ground truth labels
        probs: Predicted probabilities
        
    Returns:
        Dictionary with P@K for various K and base rate
    """
    metrics = {}
    
    # Compute P@K for standard K values
    for k in [10, 20, 50, 100, 200]:
        if k <= len(y_true):
            metrics[f'p@{k}'] = compute_precision_at_k(y_true, probs, k)
    
    # Base rate for comparison
    metrics['base_rate'] = y_true.mean()
    
    # Compute lift at K=100
    if 100 <= len(y_true):
        p100 = metrics.get('p@100', 0)
        metrics['lift@100'] = p100 / y_true.mean() if y_true.mean() > 0 else 0
    
    return metrics


def find_optimal_threshold_f05(
    y_true: np.ndarray, 
    y_proba: np.ndarray,
    min_recall: float = 0.10
) -> Dict:
    """
    Find threshold that maximizes F0.5 while maintaining minimum recall.
    
    Vela's guidance: "Maintain recall at at least 10% and maximise precision"
    
    Args:
        y_true: Binary ground truth labels
        y_proba: Predicted probabilities
        min_recall: Minimum recall constraint
        
    Returns:
        Dictionary with optimal threshold and metrics
    """
    best_f05 = 0
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        metrics = compute_precision_recall_f05(y_true, y_pred)
        
        if metrics['recall'] >= min_recall and metrics['f05'] > best_f05:
            best_f05 = metrics['f05']
            best_threshold = threshold
            best_metrics = metrics
    
    if best_metrics is None:
        y_pred = (y_proba >= 0.5).astype(int)
        best_metrics = compute_precision_recall_f05(y_true, y_pred)
        best_threshold = 0.5
    
    best_metrics['threshold'] = best_threshold
    return best_metrics


def find_threshold_max_precision(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    min_recall: float = 0.0
) -> Dict:
    """
    Find threshold that maximizes precision with optional recall constraint.
    
    Args:
        y_true: Binary ground truth labels
        y_proba: Predicted probabilities
        min_recall: Minimum recall constraint
        
    Returns:
        Dictionary with optimal threshold and metrics
    """
    best_prec = 0.0
    best_thr = 0.5
    best_metrics = None

    for thr in np.arange(0.1, 0.95, 0.01):
        y_pred = (y_proba >= thr).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_thr = thr
            best_metrics = {
                "precision": prec,
                "recall": rec,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "threshold": thr,
            }

    if best_metrics is None:
        thr = 0.5
        y_pred = (y_proba >= thr).astype(int)
        m = compute_precision_recall_f05(y_true, y_pred)
        m["threshold"] = thr
        return m

    return best_metrics


def print_ranking_metrics(
    y_true: np.ndarray, 
    probs: np.ndarray, 
    title: str = "MODEL"
) -> Dict:
    """
    Print comprehensive ranking metrics.
    
    Args:
        y_true: Binary ground truth labels
        probs: Predicted probabilities
        title: Title for the metrics display
        
    Returns:
        Dictionary with computed metrics
    """
    metrics = compute_all_ranking_metrics(y_true, probs)
    
    print(f"\n{'='*70}")
    print(f"{title} RESULTS (Ranking Metrics)")
    print(f"{'='*70}")
    print(f"Base rate: {metrics['base_rate']:.1%}")
    print(f"\nPrecision at K:")
    
    for k in [10, 20, 50, 100, 200]:
        key = f'p@{k}'
        if key in metrics:
            lift = metrics[key] / metrics['base_rate'] if metrics['base_rate'] > 0 else 0
            print(f"  P@{k:3d}: {metrics[key]:.1%} (lift: {lift:.2f}x)")
    
    if 'lift@100' in metrics:
        print(f"\nOverall lift@100: {metrics['lift@100']:.2f}x better than random")
    
    return metrics


def print_vela_metrics(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    model_name: str = "Model"
) -> Dict:
    """
    Print metrics in Vela's preferred format with benchmark comparisons.
    
    Args:
        y_true: Binary ground truth labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        
    Returns:
        Dictionary with optimal threshold metrics
    """
    base_rate = y_true.mean()
    n_positive = int(y_true.sum())
    
    print(f"\n{'='*65}")
    print(f"📊 {model_name} - VELA METRICS")
    print(f"{'='*65}")
    print(f"Base rate: {base_rate:.2%} ({n_positive} positive / {len(y_true)} total)")
    
    # P@K metrics
    print(f"\n📈 Precision @ K:")
    for k in [50, 100, 200, 500]:
        if k <= len(y_true):
            p_k = compute_precision_at_k(y_true, y_proba, k)
            lift = p_k / base_rate if base_rate > 0 else 0
            print(f"   P@{k}: {p_k:.4f} ({lift:.2f}x lift)")
    
    # Optimal threshold metrics
    print(f"\n🎯 Optimal Threshold (min recall = 0%):")
    opt = find_optimal_threshold_f05(y_true, y_proba, min_recall=0.0)
    print(f"   Threshold: {opt['threshold']:.2f}")
    print(f"   Precision: {opt['precision']:.4f} ({opt['precision']/base_rate:.2f}x lift)")
    print(f"   Recall:    {opt['recall']:.4f} {'✓' if opt['recall'] >= 0.10 else '⚠️ < 10%'}")
    print(f"   F0.5:      {opt['f05']:.4f}")
    print(f"   (TP={opt['tp']}, FP={opt['fp']}, FN={opt['fn']})")
    
    # Benchmark comparison
    print(f"\n📊 Comparison to Vela Benchmarks:")
    print(f"   {'Model':<20} {'Precision':<12} {'Recall':<10} {'F0.5':<10}")
    print(f"   {'-'*52}")
    print(f"   {'Your Model':<20} {opt['precision']:<12.3f} {opt['recall']:<10.3f} {opt['f05']:<10.3f}")
    print(f"   {'RRF (paper)':<20} {'0.131':<12} {'0.101':<10} {'0.124':<10}")
    print(f"   {'GPTree (paper)':<20} {'0.373':<12} {'0.271':<10} {'0.334':<10}")
    print(f"   {'Tier-1 VCs':<20} {'0.056':<12} {'-':<10} {'-':<10}")
    
    return opt
