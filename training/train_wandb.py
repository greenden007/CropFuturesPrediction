"""
Weights & Biases Integration for Crop Futures Training

This script wraps the training loop to work with wandb sweeps.
It reads hyperparameters from wandb.config and logs metrics.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train import train_model, EarlyStopping, train_epoch, evaluate
from dataset import load_unified_data, get_feature_groups, create_dataloaders

# Import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def train_with_wandb():
    """Training function that integrates with wandb."""
    
    # Initialize wandb if available
    if WANDB_AVAILABLE:
        wandb.init()
        config = wandb.config
    else:
        # Parse command line args if wandb not available
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='attention_vision_dual_stream')
        parser.add_argument('--commodity', type=str, default='corn')
        parser.add_argument('--horizon', type=int, default=1)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--seq_len', type=int, default=20)
        parser.add_argument('--no_early_stopping', action='store_true')
        parser.add_argument('--output_dir', type=str, default='training/results_wandb')
        parser.add_argument('--data', type=str, default='merged_data/daily_unified.csv')
        args = parser.parse_args()
        config = args
    
    # Setup paths
    data_path = getattr(config, 'data', 'merged_data/daily_unified.csv')
    output_dir = Path(getattr(config, 'output_dir', 'training/results_wandb'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target column
    commodity = getattr(config, 'commodity', 'corn')
    target_map = {'corn': 'cor_close', 'soybeans': 'soy_close', 'wheat': 'whe_close'}
    target_col = target_map.get(commodity, f'{commodity[:3]}_close')
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_unified_data(data_path)
    
    # Get feature groups
    groups = get_feature_groups(df)
    all_features = sum(groups.values(), [])
    feature_cols = [c for c in all_features if c != target_col]
    
    # Determine model type
    model_type = getattr(config, 'model', 'attention_vision_dual_stream')
    
    # Generate unique run ID to prevent checkpoint collisions
    import uuid
    run_id = str(uuid.uuid4())[:8]
    if WANDB_AVAILABLE and wandb.run:
        run_id = wandb.run.id[:8]
    
    # Build config dict for train_model
    model_config = {
        'model_type': model_type,
        'commodity': commodity,
        'target_col': target_col,
        'model_name': f"{model_type}_{commodity}_h{getattr(config, 'horizon', 1)}_sweep_{run_id}",
        'seq_len': getattr(config, 'seq_len', 20),
        'pred_horizon': getattr(config, 'horizon', 1),
        'hidden_dim': getattr(config, 'hidden_dim', 128),
        'num_layers': getattr(config, 'num_layers', 2),
        'dropout': getattr(config, 'dropout', 0.2),
        'num_epochs': getattr(config, 'epochs', 100),
        'batch_size': getattr(config, 'batch_size', 32),
        'lr': getattr(config, 'lr', 0.001),
        'train_split': 0.7,
        'val_split': 0.15,
        'early_stopping_patience': 15,
        'no_early_stopping': getattr(config, 'no_early_stopping', True),
        'weight_decay': 1e-5
    }
    
    # Vision model paths
    outlook_index_path = None
    data_dir = None
    if model_type in ['vision_dual_stream_lstm', 'attention_vision_dual_stream']:
        outlook_index_path = 'processed_data/weather_outlooks/weather_outlook_index.csv'
        data_dir = 'data/noaa_cpc_discussions/cpc_outlook_data'
        print(f"Vision model detected. Using outlook index: {outlook_index_path}")
        print(f"Data directory: {data_dir}")
        # Add to model_config so train_model receives them
        model_config['outlook_index_path'] = outlook_index_path
        model_config['data_dir'] = data_dir
    
    # Log config to wandb
    if WANDB_AVAILABLE and wandb.run:
        wandb.config.update(model_config)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    for k, v in model_config.items():
        if k not in ['predictions', 'targets']:  # Skip large arrays
            print(f"  {k}: {v}")
    
    try:
        # Train the model
        results = train_model(model_config, df, feature_cols, target_col, output_dir)
        
        # Log final metrics to wandb
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({
                'final_test_mse': results['test_metrics']['mse'],
                'final_test_mae': results['test_metrics']['mae'],
                'final_test_directional_accuracy': results['test_metrics']['directional_acc'],
                'best_epoch': results['best_epoch'],
                'final_val_dir_acc': results.get('best_val_dir_acc', 0)
            })
            
            # Log model parameters count
            wandb.log({'n_parameters': results.get('n_parameters', 0)})
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final Test Directional Accuracy: {results['test_metrics']['directional_acc']:.4f}")
        print(f"{'='*60}")
        
        # Save sweep summary
        summary = {
            'config': model_config,
            'test_directional_accuracy': float(results['test_metrics']['directional_acc']),
            'test_mse': float(results['test_metrics']['mse']),
            'test_mae': float(results['test_metrics']['mae']),
            'best_epoch': results['best_epoch']
        }
        
        summary_path = output_dir / f"{model_config['model_name']}_sweep_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return results['test_metrics']['directional_acc']
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({'error': str(e)})
        
        # Return poor performance so sweep skips this config
        return 0.0


if __name__ == "__main__":
    # If running with wandb, this will be called by wandb agent
    # If running standalone, parse args normally
    train_with_wandb()
