"""
Heterogeneous Graph Neural Network for founder success prediction.

This module defines the GNN architecture using PyTorch Geometric's
HeteroConv layers for message passing on heterogeneous graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GATConv
from typing import Dict, Tuple, Optional


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for founder success prediction.
    
    Architecture:
    1. Project each node type to hidden_dim
    2. Apply HeteroConv with SAGEConv per edge type
    3. Optional residual connections
    4. Final classification head for founder nodes
    
    Args:
        in_channels: Dictionary mapping node types to input dimensions
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        in_channels: Dict[str, int], 
        hidden_dim: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Initial projections for each node type
        self.projections = nn.ModuleDict({
            node_type: Linear(dim, hidden_dim)
            for node_type, dim in in_channels.items()
        })
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('founder', 'studied_at', 'university'): SAGEConv(hidden_dim, hidden_dim),
                ('founder', 'worked_at', 'company_size'): SAGEConv(hidden_dim, hidden_dim),
                ('founder', 'in', 'industry'): SAGEConv(hidden_dim, hidden_dim),
                ('founder', 'had', 'role_type'): SAGEConv(hidden_dim, hidden_dim),
                # Reverse edges for message passing back to founders
                ('university', 'rev_studied_at', 'founder'): SAGEConv(hidden_dim, hidden_dim),
                ('company_size', 'rev_worked_at', 'founder'): SAGEConv(hidden_dim, hidden_dim),
                ('industry', 'rev_in', 'founder'): SAGEConv(hidden_dim, hidden_dim),
                ('role_type', 'rev_had', 'founder'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='mean')
            self.convs.append(conv)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            data: HeteroData object with node features and edge indices
            
        Returns:
            Tuple of (logits, founder_embeddings)
        """
        # Project all node types to hidden dimension
        x_dict = {}
        for node_type in self.projections:
            if node_type in data.node_types:
                x_dict[node_type] = self.projections[node_type](data[node_type].x)
        
        # Build edge_index dict with reverse edges
        edge_index_dict = self._build_edge_dict(data)
        
        # Message passing through GNN layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                k: F.relu(F.dropout(v, p=self.dropout, training=self.training)) 
                for k, v in x_dict.items()
            }
        
        # Get founder embeddings and classify
        founder_emb = x_dict['founder']
        logits = self.classifier(founder_emb).squeeze(-1)
        
        return logits, founder_emb
    
    def _build_edge_dict(self, data: HeteroData) -> Dict:
        """Build edge dictionary with reverse edges."""
        edge_index_dict = {}
        
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index
            
            # Add reverse edges
            src_type, rel, dst_type = edge_type
            if src_type != dst_type:
                rev_edge_type = (dst_type, f'rev_{rel}', src_type)
                edge_index_dict[rev_edge_type] = data[edge_type].edge_index.flip(0)
        
        return edge_index_dict


class FounderGNNAdvanced(nn.Module):
    """
    Advanced GNN with residual connections and LayerNorm.
    
    This is the production model with better training dynamics.
    
    Args:
        in_channels: Dictionary mapping node types to input dimensions
        hidden_dim: Hidden layer dimension  
        num_layers: Number of GNN layers
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        in_channels: Dict[str, int], 
        hidden_dim: int = 256, 
        num_layers: int = 6, 
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Projection layers with LayerNorm
        self.projections = nn.ModuleDict({
            nt: nn.Sequential(
                Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for nt, dim in in_channels.items()
        })
        
        # GNN convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                ('founder', 'studied_at', 'university'): SAGEConv(hidden_dim, hidden_dim),
                ('founder', 'worked_at', 'company_size'): SAGEConv(hidden_dim, hidden_dim),
                ('founder', 'in', 'industry'): SAGEConv(hidden_dim, hidden_dim),
                ('founder', 'had', 'role_type'): SAGEConv(hidden_dim, hidden_dim),
                ('university', 'rev_studied_at', 'founder'): SAGEConv(hidden_dim, hidden_dim),
                ('company_size', 'rev_worked_at', 'founder'): SAGEConv(hidden_dim, hidden_dim),
                ('industry', 'rev_in', 'founder'): SAGEConv(hidden_dim, hidden_dim),
                ('role_type', 'rev_had', 'founder'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='mean'))
        
        # Deep classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connections."""
        # Project node features
        x_dict = {
            nt: self.projections[nt](data[nt].x) 
            for nt in self.projections if nt in data.node_types
        }
        
        # Build edge dictionary with reverse edges
        edge_index_dict = {}
        for et in data.edge_types:
            edge_index_dict[et] = data[et].edge_index
            src, rel, dst = et
            if src != dst:
                edge_index_dict[(dst, f'rev_{rel}', src)] = data[et].edge_index.flip(0)
        
        # Message passing with residual connections
        for i, conv in enumerate(self.convs):
            x_prev = x_dict
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                k: F.dropout(F.gelu(v), p=self.dropout, training=self.training) 
                for k, v in x_dict.items()
            }
            # Add residual connection after first layer
            if i > 0:
                x_dict = {k: v + x_prev.get(k, 0) for k, v in x_dict.items()}
        
        # Classify founders
        founder_emb = x_dict['founder']
        logits = self.classifier(founder_emb).squeeze(-1)
        
        return logits, founder_emb


def create_gnn_model(
    in_channels: Dict[str, int],
    hidden_dim: int = 256,
    num_layers: int = 6,
    dropout: float = 0.4,
    device: str = 'cpu',
    use_advanced: bool = True
) -> nn.Module:
    """
    Factory function to create GNN model.
    
    Args:
        in_channels: Dictionary mapping node types to input dimensions
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout probability
        device: Target device
        use_advanced: Whether to use advanced model with residuals
        
    Returns:
        GNN model on specified device
    """
    if use_advanced:
        model = FounderGNNAdvanced(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        model = HeteroGNN(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    return model.to(device)
