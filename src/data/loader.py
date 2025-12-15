"""
Data loading utilities for founder datasets.

This module handles loading and combining public and private datasets
for founder success prediction.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional


def load_and_combine_datasets(
    public_path: Union[str, Path],
    private_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load and combine public + private founder datasets.
    
    Args:
        public_path: Path to public dataset CSV
        private_path: Optional path to private dataset CSV
        verbose: Whether to print loading statistics
        
    Returns:
        Combined DataFrame with 'source' column indicating origin
    """
    if verbose:
        print("=" * 50)
        print("LOADING & COMBINING DATASETS")
        print("=" * 50)
    
    # Load public dataset
    df_public = pd.read_csv(public_path)
    df_public['source'] = 'public'
    
    if verbose:
        print(f"Public:  {len(df_public)} founders ({df_public['success'].sum()} successful)")
    
    if private_path is not None and Path(private_path).exists():
        # Load private dataset
        df_private = pd.read_csv(private_path)
        df_private['source'] = 'private'
        
        if verbose:
            print(f"Private: {len(df_private)} founders ({df_private['success'].sum()} successful)")
        
        # Check for overlaps
        public_uuids = set(df_public['founder_uuid'])
        private_uuids = set(df_private['founder_uuid'])
        overlap = len(public_uuids & private_uuids)
        
        if overlap > 0:
            if verbose:
                print(f"\n⚠️  WARNING: {overlap} founders in both datasets - removing duplicates")
            df_private = df_private[~df_private['founder_uuid'].isin(public_uuids)]
        
        # Combine datasets
        df_combined = pd.concat([df_public, df_private], ignore_index=True)
    else:
        df_combined = df_public
        if verbose and private_path is not None:
            print(f"Private dataset not found at {private_path}, using public only")
    
    if verbose:
        print(f"\nCombined: {len(df_combined)} founders")
        print(f"  Success rate: {df_combined['success'].mean()*100:.2f}%")
        print(f"  Successful: {df_combined['success'].sum()}")
        print("=" * 50)
    
    return df_combined


def load_processed_data(
    baseline_features_path: Union[str, Path],
    labels_path: Union[str, Path],
    embeddings_path: Optional[Union[str, Path]] = None,
    predictions_path: Optional[Union[str, Path]] = None,
    masks_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> dict:
    """
    Load pre-processed data files.
    
    Args:
        baseline_features_path: Path to baseline features CSV
        labels_path: Path to labels CSV
        embeddings_path: Optional path to GNN embeddings NPY
        predictions_path: Optional path to GNN predictions NPY
        masks_path: Optional path to train/val/test masks CSV
        verbose: Whether to print loading statistics
        
    Returns:
        Dictionary containing loaded data arrays
    """
    import numpy as np
    
    data = {}
    
    # Load baseline features
    data['baseline_features'] = pd.read_csv(baseline_features_path)
    if verbose:
        print(f"✓ Loaded baseline features: {data['baseline_features'].shape}")
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    data['labels'] = labels_df['success'].values
    if verbose:
        print(f"✓ Loaded labels: {len(data['labels'])} founders")
    
    # Load embeddings if provided
    if embeddings_path is not None and Path(embeddings_path).exists():
        data['embeddings'] = np.load(embeddings_path)
        if verbose:
            print(f"✓ Loaded embeddings: {data['embeddings'].shape}")
    
    # Load predictions if provided
    if predictions_path is not None and Path(predictions_path).exists():
        data['predictions'] = np.load(predictions_path)
        if verbose:
            print(f"✓ Loaded predictions: {data['predictions'].shape}")
    
    # Load masks if provided
    if masks_path is not None and Path(masks_path).exists():
        masks_df = pd.read_csv(masks_path)
        data['train_mask'] = masks_df['train_mask'].values.astype(bool)
        data['val_mask'] = masks_df['val_mask'].values.astype(bool)
        data['test_mask'] = masks_df['test_mask'].values.astype(bool)
        if verbose:
            print(f"✓ Loaded masks: train={data['train_mask'].sum()}, "
                  f"val={data['val_mask'].sum()}, test={data['test_mask'].sum()}")
    
    return data
