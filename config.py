"""
Configuration settings for the Founder Success Prediction pipeline.

This module centralizes all configurable parameters including paths,
model hyperparameters, and training settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class PathConfig:
    """Data and output path configurations."""
    # Input paths
    data_dir: Path = Path("data/raw")
    public_data: str = "vcbench_final_public.csv"
    private_data: str = "vcbench_final_private.csv"
    
    # Output paths
    output_dir: Path = Path("outputs")
    models_dir: Path = Path("outputs/models")
    plots_dir: Path = Path("outputs/plots")
    results_dir: Path = Path("outputs/results")
    processed_dir: Path = Path("data/processed")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.output_dir, self.models_dir, self.plots_dir, 
                         self.results_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """GNN model architecture configuration."""
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.4
    aggregation: str = "mean"
    
    # Node types in the heterogeneous graph
    node_types: List[str] = field(default_factory=lambda: [
        'founder', 'university', 'company_size', 'industry', 'role_type'
    ])
    
    # Edge types in the heterogeneous graph
    edge_types: List[tuple] = field(default_factory=lambda: [
        ('founder', 'studied_at', 'university'),
        ('founder', 'worked_at', 'company_size'),
        ('founder', 'in', 'industry'),
        ('founder', 'had', 'role_type'),
    ])


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Training parameters
    epochs: int = 2000
    learning_rate: float = 0.0005
    weight_decay: float = 1e-3
    warmup_epochs: int = 100
    
    # Loss settings
    precision_weight: float = 10.0  # FP cost multiplier
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Evaluation
    target_k: int = 100  # Optimize for P@100
    eval_every: int = 100
    
    # Multi-seed ensemble
    n_seeds: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SplitConfig:
    """Train/validation/test split configuration."""
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42


@dataclass
class ClusteringConfig:
    """Clustering configuration."""
    min_clusters: int = 2
    max_clusters: int = 10
    n_init: int = 20
    optimal_k: int = 4  # Can be auto-determined
    random_seed: int = 42


@dataclass
class Config:
    """Main configuration container."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    
    # Experiment settings
    experiment_name: str = "founder_gnn_experiment"
    verbose: bool = True
    save_checkpoints: bool = True


def get_default_config() -> Config:
    """Return default configuration."""
    return Config()


def get_config_from_args(args) -> Config:
    """Create configuration from command-line arguments."""
    config = get_default_config()
    
    # Override with args if provided
    if hasattr(args, 'epochs'):
        config.training.epochs = args.epochs
    if hasattr(args, 'hidden_dim'):
        config.model.hidden_dim = args.hidden_dim
    if hasattr(args, 'n_seeds'):
        config.training.n_seeds = args.n_seeds
    if hasattr(args, 'n_clusters'):
        config.clustering.optimal_k = args.n_clusters
        
    return config
