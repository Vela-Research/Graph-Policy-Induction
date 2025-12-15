"""
PyTorch Geometric HeteroData construction.

This module converts the graph structure into PyTorch Geometric
HeteroData format for GNN training.
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Tuple


def build_heterodata(
    entities: Dict, 
    edges: Dict, 
    baseline_features: pd.DataFrame,
    verbose: bool = True
) -> HeteroData:
    """
    Convert graph structure to PyTorch Geometric HeteroData.
    
    Args:
        entities: Dictionary with node types and their attributes
        edges: Dictionary with edge types and their connections
        baseline_features: DataFrame with founder baseline features
        verbose: Whether to print construction progress
        
    Returns:
        HeteroData object ready for GNN training
    """
    if verbose:
        print("\n" + "="*50)
        print("BUILDING HETERODATA")
        print("="*50)
    
    data = HeteroData()
    
    # Add founder nodes with features and labels
    founder_features = torch.tensor(
        baseline_features.fillna(0).values, 
        dtype=torch.float
    )
    data['founder'].x = founder_features
    data['founder'].y = torch.tensor(entities['founder']['labels'], dtype=torch.float)
    
    if verbose:
        print(f"  founder: {founder_features.shape[0]} nodes, {founder_features.shape[1]} features")
    
    # Add other node types with one-hot identity features
    for node_type in ['university', 'company_size', 'industry', 'role_type']:
        n = entities[node_type]['count']
        data[node_type].x = torch.eye(n, dtype=torch.float)
        if verbose:
            print(f"  {node_type}: {n} nodes")
    
    # Add edges
    edge_mapping = {
        'founder_studied_at_university': ('founder', 'studied_at', 'university'),
        'founder_worked_at_company': ('founder', 'worked_at', 'company_size'),
        'founder_in_industry': ('founder', 'in', 'industry'),
        'founder_had_role': ('founder', 'had', 'role_type'),
    }
    
    if verbose:
        print("\nEdges:")
    
    for edge_name, edge_type in edge_mapping.items():
        if edge_name in edges:
            src, tgt = edges[edge_name]
            if len(src) > 0:
                data[edge_type].edge_index = torch.tensor([src, tgt], dtype=torch.long)
                if verbose:
                    print(f"  {edge_type}: {len(src)} edges")
    
    return data


def get_input_channels(data: HeteroData) -> Dict[str, int]:
    """
    Get input channel dimensions for each node type.
    
    Args:
        data: HeteroData object
        
    Returns:
        Dictionary mapping node types to their feature dimensions
    """
    return {nt: data[nt].x.shape[1] for nt in data.node_types}


def add_reverse_edges(data: HeteroData) -> Dict:
    """
    Create reverse edge dictionary for bidirectional message passing.
    
    Args:
        data: HeteroData object
        
    Returns:
        Dictionary with original and reverse edges
    """
    edge_index_dict = {}
    
    for edge_type in data.edge_types:
        edge_index_dict[edge_type] = data[edge_type].edge_index
        
        # Add reverse edges
        src_type, rel, dst_type = edge_type
        if src_type != dst_type:
            rev_edge_type = (dst_type, f'rev_{rel}', src_type)
            edge_index_dict[rev_edge_type] = data[edge_type].edge_index.flip(0)
    
    return edge_index_dict


def heterodata_to_device(data: HeteroData, device: str) -> HeteroData:
    """
    Move HeteroData to specified device.
    
    Args:
        data: HeteroData object
        device: Target device ('cuda' or 'cpu')
        
    Returns:
        HeteroData on target device
    """
    return data.to(device)


def save_heterodata(data: HeteroData, path: str):
    """
    Save HeteroData object to disk.
    
    Args:
        data: HeteroData object
        path: Save path
    """
    torch.save(data, path)


def load_heterodata(path: str, device: str = 'cpu') -> HeteroData:
    """
    Load HeteroData object from disk.
    
    Args:
        path: Load path
        device: Target device
        
    Returns:
        Loaded HeteroData object
    """
    data = torch.load(path, map_location=device)
    return data
