#!/usr/bin/env python3
"""
Founder Success Prediction using Graph Neural Networks

Main entry point for the pipeline. This script orchestrates:
1. Data loading and preprocessing
2. Graph construction
3. GNN training
4. Clustering
5. Test set evaluation
6. Visualization

Usage:
    python main.py --epochs 2000 --n-seeds 10 --n-clusters 4
    python main.py --mode train
    python main.py --mode evaluate --model-path outputs/models/best_model.pt
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import Config, get_default_config
from src.data import (
    load_and_combine_datasets,
    extract_education_features,
    extract_job_features,
    remove_redundant_features,
    create_train_val_test_masks
)
from src.graph import build_graph, build_heterodata
from src.models import (
    train_multiseed_ensemble,
    get_predictions,
    get_embeddings,
    select_best_model
)
from src.clustering import (
    find_optimal_clusters,
    perform_clustering,
    analyze_clusters,
    save_cluster_results
)
from src.evaluation import (
    evaluate_test_set,
    assign_test_to_clusters,
    analyze_cluster_performance,
    where_do_successful_founders_go
)
from src.visualization import (
    plot_cluster_optimization,
    plot_cluster_visualization,
    plot_test_evaluation,
    plot_training_history,
    plot_precision_at_k
)

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Founder Success Prediction using GNNs'
    )
    
    # Mode
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'train', 'evaluate', 'cluster'],
                        help='Pipeline mode')
    
    # Data paths
    parser.add_argument('--public-data', type=str, 
                        default='data/raw/vcbench_final_public.csv',
                        help='Path to public dataset')
    parser.add_argument('--private-data', type=str,
                        default='data/raw/vcbench_final_private.csv',
                        help='Path to private dataset')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for GNN')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds for ensemble')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--precision-weight', type=float, default=10.0,
                        help='FP cost multiplier in loss')
    parser.add_argument('--target-k', type=int, default=100,
                        help='K value for P@K optimization')
    
    # Clustering
    parser.add_argument('--n-clusters', type=int, default=4,
                        help='Number of clusters')
    parser.add_argument('--auto-clusters', action='store_true',
                        help='Automatically determine optimal clusters')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--experiment-name', type=str, default='experiment',
                        help='Name for this experiment')
    
    # Other
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup and return device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    print(f"Using device: {device}")
    return device


def setup_output_dirs(output_dir: str, experiment_name: str) -> dict:
    """Create and return output directory paths."""
    base_dir = Path(output_dir) / experiment_name
    dirs = {
        'base': base_dir,
        'models': base_dir / 'models',
        'plots': base_dir / 'plots',
        'results': base_dir / 'results',
        'processed': base_dir / 'processed'
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run_full_pipeline(args):
    """Run the complete pipeline."""
    
    # Setup
    device = setup_device(args.device)
    dirs = setup_output_dirs(args.output_dir, args.experiment_name)
    
    print("\n" + "="*80)
    print("FOUNDER SUCCESS PREDICTION - FULL PIPELINE")
    print("="*80)
    
    # =========================================================================
    # STEP 1: Data Loading & Preprocessing
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Data Loading & Preprocessing")
    print("="*80)
    
    # Load data
    df = load_and_combine_datasets(
        args.public_data,
        args.private_data if Path(args.private_data).exists() else None,
        verbose=args.verbose
    )
    
    # Extract features
    edu_features = extract_education_features(df, verbose=args.verbose)
    job_features = extract_job_features(df, verbose=args.verbose)
    
    X_baseline = pd.concat([edu_features, job_features], axis=1)
    y = df['success']
    
    print(f"\nTotal baseline features: {len(X_baseline.columns)}")
    
    # Remove redundant features
    X_baseline_clean, _ = remove_redundant_features(X_baseline, y, threshold=0.6)
    
    # Save processed data
    X_baseline_clean.to_csv(dirs['processed'] / 'baseline_features.csv', index=False)
    pd.DataFrame({'success': y}).to_csv(dirs['processed'] / 'labels.csv', index=False)
    
    # =========================================================================
    # STEP 2: Graph Construction
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Graph Construction")
    print("="*80)
    
    entities, edges = build_graph(df, verbose=args.verbose)
    data = build_heterodata(entities, edges, X_baseline_clean, verbose=args.verbose)
    data = data.to(device)
    
    # Create train/val/test split
    data = create_train_val_test_masks(
        data,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_seed=42,
        save_path=dirs['processed'] / 'masks.npz',
        verbose=args.verbose
    )
    
    # =========================================================================
    # STEP 3: GNN Training
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: GNN Training")
    print("="*80)
    
    models, embeddings, results = train_multiseed_ensemble(
        data,
        device=device,
        n_seeds=args.n_seeds,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        precision_weight=args.precision_weight,
        target_k=args.target_k,
        verbose=args.verbose
    )
    
    # Select and save best model
    best_model, best_idx, best_pk = select_best_model(
        models, data, device, args.target_k, verbose=args.verbose
    )
    
    torch.save(best_model.state_dict(), dirs['models'] / 'best_model.pt')
    
    # Save ensemble
    ensemble_dir = dirs['models'] / 'ensemble'
    ensemble_dir.mkdir(exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), ensemble_dir / f'model_seed_{i}.pt')
    
    # Get predictions and embeddings
    predictions = get_predictions(models, data, device)
    embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
    
    np.save(dirs['results'] / 'predictions.npy', predictions)
    np.save(dirs['results'] / 'embeddings.npy', embeddings_np)
    
    # Save results
    with open(dirs['results'] / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Plot training history if available
    if not args.no_plots:
        plot_precision_at_k(
            results, 
            results['base_rate']['mean'],
            model_name="GNN Ensemble",
            save_path=str(dirs['plots'] / 'precision_at_k.png'),
            show=False
        )
    
    # =========================================================================
    # STEP 4: Clustering
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Clustering")
    print("="*80)
    
    # Get train+val data only
    train_val_mask = (data['founder'].train_mask | data['founder'].val_mask).cpu().numpy()
    y_train_val = y.values[train_val_mask]
    X_baseline_train_val = X_baseline_clean[train_val_mask].reset_index(drop=True)
    embeddings_train_val = embeddings_np[train_val_mask]
    predictions_train_val = predictions[train_val_mask]
    
    # Find optimal clusters if requested
    if args.auto_clusters:
        combined_temp = np.concatenate([
            (embeddings_train_val - embeddings_train_val.mean(0)) / (embeddings_train_val.std(0) + 1e-8),
            (X_baseline_train_val.fillna(0).values - X_baseline_train_val.fillna(0).values.mean(0)) / (X_baseline_train_val.fillna(0).values.std(0) + 1e-8)
        ], axis=1)
        optimal_k, silhouette_scores = find_optimal_clusters(
            combined_temp, verbose=args.verbose
        )
        
        if not args.no_plots:
            plot_cluster_optimization(
                range(2, 11), silhouette_scores, optimal_k,
                save_path=str(dirs['plots'] / 'cluster_optimization.png'),
                show=False
            )
    else:
        optimal_k = args.n_clusters
    
    # Perform clustering
    clusters, kmeans, scaler_emb, scaler_base, combined_train = perform_clustering(
        embeddings_train_val,
        X_baseline_train_val,
        n_clusters=optimal_k,
        verbose=args.verbose
    )
    
    # Analyze clusters
    cluster_descriptions, cluster_stats = analyze_clusters(
        clusters, y_train_val, predictions_train_val,
        X_baseline_train_val, optimal_k, verbose=args.verbose
    )
    
    # Save cluster results
    save_cluster_results(
        clusters, cluster_descriptions, predictions_train_val,
        y_train_val, str(dirs['results']), verbose=args.verbose
    )
    
    # Plot cluster visualization
    if not args.no_plots:
        plot_cluster_visualization(
            combined_train, clusters, y_train_val, cluster_descriptions,
            save_path=str(dirs['plots'] / 'cluster_visualization.png'),
            show=False
        )
    
    # =========================================================================
    # STEP 5: Test Set Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Test Set Evaluation")
    print("="*80)
    
    # Get test data
    test_mask = data['founder'].test_mask.cpu().numpy()
    y_test = y.values[test_mask]
    X_baseline_test = X_baseline_clean[test_mask].reset_index(drop=True)
    embeddings_test = embeddings_np[test_mask]
    predictions_test = predictions[test_mask]
    
    # Assign test founders to clusters
    clusters_test = assign_test_to_clusters(
        embeddings_test, X_baseline_test.fillna(0).values,
        scaler_emb, scaler_base, kmeans, verbose=args.verbose
    )
    
    # Analyze cluster performance
    cluster_success_rates, _ = analyze_cluster_performance(
        y_train_val, clusters, y_test, clusters_test, optimal_k, verbose=args.verbose
    )
    
    # Where do successful founders go?
    distribution = where_do_successful_founders_go(
        y_test, clusters_test, cluster_success_rates, verbose=args.verbose
    )
    
    # Full evaluation
    test_results = evaluate_test_set(
        y_test, predictions_test, clusters_test, cluster_success_rates,
        verbose=args.verbose
    )
    
    # Save test results
    with open(dirs['results'] / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=float)
    
    # Plot test evaluation
    if not args.no_plots:
        plot_test_evaluation(
            combined_train, 
            np.concatenate([scaler_emb.transform(embeddings_test), 
                           scaler_base.transform(X_baseline_test.fillna(0).values)], axis=1),
            clusters, clusters_test,
            y_train_val, y_test,
            cluster_success_rates, optimal_k,
            save_path=str(dirs['plots'] / 'test_evaluation.png'),
            show=False
        )
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {dirs['base']}")
    print(f"  Models:  {dirs['models']}")
    print(f"  Plots:   {dirs['plots']}")
    print(f"  Results: {dirs['results']}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"  Founders: {len(y)}")
    print(f"  Success rate: {y.mean():.1%}")
    print(f"  GNN P@{args.target_k}: {results[f'p@{args.target_k}']['mean']:.1%}")
    print(f"  Clusters: {optimal_k}")
    
    return {
        'models': models,
        'embeddings': embeddings_np,
        'predictions': predictions,
        'clusters': clusters,
        'cluster_descriptions': cluster_descriptions,
        'test_results': test_results
    }


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'full':
        run_full_pipeline(args)
    elif args.mode == 'train':
        # Training only mode - would need separate implementation
        print("Training-only mode not yet implemented. Use --mode full")
    elif args.mode == 'evaluate':
        # Evaluation only mode - would need separate implementation
        print("Evaluation-only mode not yet implemented. Use --mode full")
    elif args.mode == 'cluster':
        # Clustering only mode - would need separate implementation
        print("Clustering-only mode not yet implemented. Use --mode full")


if __name__ == '__main__':
    main()
