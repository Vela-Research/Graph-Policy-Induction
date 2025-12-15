"""
GNN training utilities for founder success prediction.

This module provides training functions optimized for ranking metrics
(P@K) rather than traditional classification metrics.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import math
from typing import Dict, Tuple, Optional, List
from torch_geometric.data import HeteroData

from .gnn import create_gnn_model
from .losses import get_precision_weighted_loss
from ..evaluation.metrics import compute_precision_at_k, compute_all_ranking_metrics


def train_gnn_ranking(
    data: HeteroData,
    device: str = 'cpu',
    hidden_dim: int = 256,
    num_layers: int = 6,
    dropout: float = 0.4,
    epochs: int = 600,
    lr: float = 0.0005,
    weight_decay: float = 1e-3,
    warmup_epochs: int = 100,
    precision_weight: float = 5.0,
    target_k: int = 100,
    eval_every: int = 50,
    verbose: bool = True
) -> Tuple[nn.Module, torch.Tensor, Dict]:
    """
    Train GNN optimizing for P@K (ranking).
    
    This training function uses ranking-based evaluation (P@K) rather than
    threshold-based metrics. The model learns to rank founders by their
    probability of success.
    
    Args:
        data: HeteroData with founder nodes, features, and labels
        device: Training device ('cuda' or 'cpu')
        hidden_dim: Hidden dimension for GNN layers
        num_layers: Number of GNN layers
        dropout: Dropout probability
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs for learning rate
        precision_weight: FP cost multiplier in loss function
        target_k: K value for P@K optimization
        eval_every: Evaluate every N epochs
        verbose: Whether to print training progress
        
    Returns:
        Tuple of (trained model, embeddings, training history)
    """
    if verbose:
        print("\n" + "="*70)
        print(f"🚀 RANKING-OPTIMIZED GNN TRAINING (P@{target_k})")
        print("="*70)
        print(f"""
    Architecture:  {hidden_dim}d, {num_layers} layers, {dropout} dropout
    Training:      {epochs} epochs, LR {lr} with cosine decay
    Loss:          Weighted BCE (FP cost {precision_weight:.1f}x more than FN)
    Optimization:  P@{target_k} (ranking)
        """)
    
    start_time = time.time()
    data = data.to(device)
    
    # Create model
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = create_gnn_model(
        in_channels, hidden_dim, num_layers, dropout, device
    )
    
    # Loss function
    criterion = get_precision_weighted_loss(precision_weight, device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Learning rate schedule with warmup
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 1e-7 + 0.5 * (lr - 1e-7) * (1 + math.cos(math.pi * progress))
    
    # Training state
    best_val_pk = 0
    best_state = None
    best_epoch = 0
    history = {'loss': [], f'val_p@{target_k}': [], 'lr': []}
    
    if verbose:
        print(f"\n{'Epoch':<8} {'Loss':<10} {f'Val P@{target_k}':<12} {'Time'}")
        print("-"*50)
    
    # Training loop
    for epoch in range(epochs):
        # Update learning rate
        current_lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits, _ = model(data)
        train_mask = data['founder'].train_mask
        loss = criterion(logits[train_mask], data['founder'].y[train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history['loss'].append(loss.item())
        history['lr'].append(current_lr)
        
        # Evaluation
        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits, _ = model(data)
                probs = torch.sigmoid(logits)
            
            val_mask = data['founder'].val_mask
            y_val = data['founder'].y[val_mask].cpu().numpy()
            probs_val = probs[val_mask].cpu().numpy()
            
            # Compute P@K
            val_pk = compute_precision_at_k(y_val, probs_val, min(target_k, len(y_val)))
            history[f'val_p@{target_k}'].append(val_pk)
            
            # Track best model
            if val_pk > best_val_pk:
                best_val_pk = val_pk
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                status = "★"
            else:
                status = ""
            
            if verbose:
                elapsed = (time.time() - start_time) / 60
                print(f"{epoch:<8} {loss.item():<10.4f} {val_pk:<12.4f} {elapsed:.1f}m {status}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(data)
        probs = torch.sigmoid(logits)
    
    val_mask = data['founder'].val_mask
    y_val = data['founder'].y[val_mask].cpu().numpy()
    probs_val = probs[val_mask].cpu().numpy()
    
    val_metrics = compute_all_ranking_metrics(y_val, probs_val)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE - {(time.time() - start_time)/60:.1f} minutes")
        print(f"{'='*70}")
        print(f"Best epoch: {best_epoch}")
        print(f"\nValidation Results:")
        print(f"  Base rate: {val_metrics['base_rate']:.1%}")
        
        for k in [10, 20, 50, 100, 200]:
            key = f'p@{k}'
            if key in val_metrics:
                lift = val_metrics[key] / val_metrics['base_rate']
                print(f"  P@{k:3d}: {val_metrics[key]:.1%} (lift: {lift:.2f}x)")
    
    history['val_results'] = val_metrics
    
    return model, embeddings, history


def train_multiseed_ensemble(
    data: HeteroData,
    device: str = 'cpu',
    n_seeds: int = 5,
    epochs: int = 600,
    target_k: int = 100,
    verbose: bool = True,
    **kwargs
) -> Tuple[List[nn.Module], torch.Tensor, Dict]:
    """
    Train multiple models with different seeds and ensemble.
    
    Args:
        data: HeteroData object
        device: Training device
        n_seeds: Number of random seeds to use
        epochs: Training epochs per model
        target_k: K value for P@K optimization
        verbose: Whether to print progress
        **kwargs: Additional arguments for train_gnn_ranking
        
    Returns:
        Tuple of (list of models, ensemble embeddings, averaged results)
    """
    all_models = []
    all_embeddings = []
    all_results = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"\n{'='*70}")
            print(f"SEED {seed+1}/{n_seeds}")
            print(f"{'='*70}")
        
        # Set seeds
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        # Train model
        model, embeddings, history = train_gnn_ranking(
            data, device, epochs=epochs,
            target_k=target_k,
            verbose=(seed == 0 and verbose),  # Only show progress for first seed
            **kwargs
        )
        
        all_models.append(model)
        all_embeddings.append(embeddings.cpu().numpy())
        all_results.append(history['val_results'])
    
    # Ensemble embeddings
    ensemble_embeddings = np.mean(all_embeddings, axis=0)
    
    # Average results across seeds
    avg_results = {}
    for key in all_results[0].keys():
        if isinstance(all_results[0][key], (int, float, np.integer, np.floating)):
            values = [r[key] for r in all_results]
            avg_results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ENSEMBLE RESULTS ({n_seeds} seeds)")
        print(f"{'='*70}")
        
        base_rate = avg_results['base_rate']['mean']
        print(f"\nBase rate: {base_rate:.1%}")
        print(f"\nPrecision at K (mean ± std):")
        
        for k in [10, 20, 50, 100, 200]:
            key = f'p@{k}'
            if key in avg_results:
                m = avg_results[key]
                lift = m['mean'] / base_rate
                print(f"  P@{k:3d}: {m['mean']:.1%} ± {m['std']:.1%} (lift: {lift:.2f}x)")
    
    return all_models, torch.tensor(ensemble_embeddings), avg_results


def get_predictions(
    models: List[nn.Module],
    data: HeteroData,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Get ensemble predictions from multiple models.
    
    Args:
        models: List of trained models
        data: HeteroData object
        device: Device for inference
        
    Returns:
        Averaged prediction probabilities
    """
    all_predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            logits, _ = model(data.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_predictions.append(probs)
    
    return np.mean(all_predictions, axis=0)


def get_embeddings(
    models: List[nn.Module],
    data: HeteroData,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Get ensemble embeddings from multiple models.
    
    Args:
        models: List of trained models
        data: HeteroData object
        device: Device for inference
        
    Returns:
        Averaged embeddings
    """
    all_embeddings = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            _, embeddings = model(data.to(device))
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.mean(all_embeddings, axis=0)


def select_best_model(
    models: List[nn.Module],
    data: HeteroData,
    device: str = 'cpu',
    target_k: int = 100,
    verbose: bool = True
) -> Tuple[nn.Module, int, float]:
    """
    Select the best model from ensemble based on validation P@K.
    
    Args:
        models: List of trained models
        data: HeteroData object
        device: Device for inference
        target_k: K value for P@K evaluation
        verbose: Whether to print results
        
    Returns:
        Tuple of (best model, best model index, best P@K score)
    """
    best_model_idx = 0
    best_pk = 0
    
    val_mask = data['founder'].val_mask.cpu().numpy()
    y_val = data['founder'].y.cpu().numpy()[val_mask]
    
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            logits, _ = model(data.to(device))
            preds = torch.sigmoid(logits).cpu().numpy()
        
        val_preds = preds[val_mask]
        pk = compute_precision_at_k(y_val, val_preds, min(target_k, len(y_val)))
        
        if verbose:
            print(f"  Model {i+1}: P@{target_k} = {pk:.1%}")
        
        if pk > best_pk:
            best_pk = pk
            best_model_idx = i
    
    if verbose:
        print(f"\n✓ Best model: #{best_model_idx+1} with P@{target_k} = {best_pk:.1%}")
    
    return models[best_model_idx], best_model_idx, best_pk
