"""
Train/validation/test splitting utilities.

This module handles stratified splitting of founder data
with proper mask creation for GNN training.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
from typing import Union, Optional, Dict


def create_train_val_test_masks(
    data: HeteroData,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
    save_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> HeteroData:
    """
    Create stratified train/val/test masks for founder nodes.
    
    Stratification ensures each split maintains the same success rate
    as the overall dataset.
    
    Args:
        data: HeteroData object with founder nodes and labels
        train_ratio: Fraction for training (default 0.80)
        val_ratio: Fraction for validation (default 0.10)
        test_ratio: Fraction for testing (default 0.10)
        random_seed: Random seed for reproducibility
        save_path: Optional path to save masks
        verbose: Whether to print statistics
        
    Returns:
        HeteroData with train_mask, val_mask, test_mask attached
    """
    n_founders = data['founder'].x.shape[0]
    labels = data['founder'].y.cpu().numpy()
    
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0! Got {train_ratio + val_ratio + test_ratio}"
    
    if verbose:
        print("\n" + "="*80)
        print("CREATING TRAIN/VAL/TEST SPLIT")
        print("="*80)
    
    # Get indices for positive and negative samples
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    if verbose:
        print(f"\nTotal founders: {n_founders}")
        print(f"  Positive: {len(pos_idx)} ({len(pos_idx)/n_founders*100:.1f}%)")
        print(f"  Negative: {len(neg_idx)} ({len(neg_idx)/n_founders*100:.1f}%)")
    
    # Shuffle with fixed seed
    np.random.seed(random_seed)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    # Split positive samples
    n_pos_train = int(len(pos_idx) * train_ratio)
    n_pos_val = int(len(pos_idx) * val_ratio)
    
    pos_train = pos_idx[:n_pos_train]
    pos_val = pos_idx[n_pos_train:n_pos_train + n_pos_val]
    pos_test = pos_idx[n_pos_train + n_pos_val:]
    
    # Split negative samples
    n_neg_train = int(len(neg_idx) * train_ratio)
    n_neg_val = int(len(neg_idx) * val_ratio)
    
    neg_train = neg_idx[:n_neg_train]
    neg_val = neg_idx[n_neg_train:n_neg_train + n_neg_val]
    neg_test = neg_idx[n_neg_train + n_neg_val:]
    
    # Combine and shuffle
    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])
    test_idx = np.concatenate([pos_test, neg_test])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    # Create boolean masks
    train_mask = torch.zeros(n_founders, dtype=torch.bool)
    val_mask = torch.zeros(n_founders, dtype=torch.bool)
    test_mask = torch.zeros(n_founders, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Attach to data
    device = data['founder'].y.device
    data['founder'].train_mask = train_mask.to(device)
    data['founder'].val_mask = val_mask.to(device)
    data['founder'].test_mask = test_mask.to(device)
    
    if verbose:
        print(f"\n{'='*80}")
        print("SPLIT STATISTICS")
        print("="*80)
        print(f"\nTrain: {len(train_idx)} founders ({len(train_idx)/n_founders*100:.1f}%)")
        print(f"  Successful: {labels[train_idx].sum()} ({labels[train_idx].mean():.2%})")
        print(f"\nVal:   {len(val_idx)} founders ({len(val_idx)/n_founders*100:.1f}%)")
        print(f"  Successful: {labels[val_idx].sum()} ({labels[val_idx].mean():.2%})")
        print(f"\nTest:  {len(test_idx)} founders ({len(test_idx)/n_founders*100:.1f}%)")
        print(f"  Successful: {labels[test_idx].sum()} ({labels[test_idx].mean():.2%})")
        print(f"  ⚠️  TEST SET - DO NOT USE UNTIL FINAL EVALUATION!")
    
    # Save masks if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        np.savez(
            save_path,
            train_mask=train_mask.cpu().numpy(),
            val_mask=val_mask.cpu().numpy(),
            test_mask=test_mask.cpu().numpy(),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            n_founders=n_founders,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )
        
        if verbose:
            print(f"\n✓ Saved masks to: {save_path}")
    
    return data


def load_masks(
    mask_path: Union[str, Path],
    n_founders: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Load pre-saved train/val/test masks.
    
    Args:
        mask_path: Path to saved masks NPZ file
        n_founders: Expected number of founders (for validation)
        
    Returns:
        Dictionary with mask arrays
    """
    loaded = np.load(mask_path)
    
    masks = {
        'train_mask': loaded['train_mask'],
        'val_mask': loaded['val_mask'],
        'test_mask': loaded['test_mask'],
        'train_idx': loaded['train_idx'],
        'val_idx': loaded['val_idx'],
        'test_idx': loaded['test_idx'],
    }
    
    if n_founders is not None:
        assert len(masks['train_mask']) == n_founders, \
            f"Mask length mismatch: expected {n_founders}, got {len(masks['train_mask'])}"
    
    return masks


def apply_masks_to_data(
    data: HeteroData,
    masks: Dict[str, np.ndarray]
) -> HeteroData:
    """
    Apply loaded masks to HeteroData object.
    
    Args:
        data: HeteroData object
        masks: Dictionary with mask arrays
        
    Returns:
        HeteroData with masks attached
    """
    device = data['founder'].y.device
    
    data['founder'].train_mask = torch.tensor(masks['train_mask'], dtype=torch.bool).to(device)
    data['founder'].val_mask = torch.tensor(masks['val_mask'], dtype=torch.bool).to(device)
    data['founder'].test_mask = torch.tensor(masks['test_mask'], dtype=torch.bool).to(device)
    
    return data
