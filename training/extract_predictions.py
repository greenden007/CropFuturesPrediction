"""
Extract Predictions from Saved Model Checkpoints

Loads saved model files and generates predictions for the test set,
saving them back to the results JSON for trading simulation.

Usage:
    python extract_predictions.py --model dual_stream_lstm --commodity corn --horizon 1
    python extract_predictions.py --all  # Extract for all saved models
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from dataset import load_unified_data, get_feature_groups, create_dataloaders, DualStreamDataset
from models import create_model


def load_model_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Recreate model
    model = create_model(config['model_type'], **checkpoint.get('model_kwargs', {}))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def extract_predictions(
    model_name: str,
    commodity: str,
    horizon: int,
    results_dir: Path,
    data_path: Path,
    device: torch.device
) -> Dict:
    """Extract predictions from saved model for test set."""
    
    model_file = results_dir / f"{model_name}_{commodity}_h{horizon}_best.pt"
    results_file = results_dir / f"{model_name}_{commodity}_h{horizon}_results.json"
    
    if not model_file.exists():
        print(f"Model file not found: {model_file}")
        return None
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    print(f"\nExtracting predictions for {model_name} on {commodity} (H={horizon})")
    
    # Load checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    config = checkpoint['config']
    
    # Load data
    df = load_unified_data(data_path)
    target_col = config['target_col']
    
    # Get feature columns
    groups = get_feature_groups(df)
    feature_cols = sum(groups.values(), [])
    feature_cols = [c for c in feature_cols if c != target_col]
    
    # Infer expected input dimensions from saved model weights
    state_dict = checkpoint['model_state_dict']
    
    if config['model_type'] == 'dual_stream_lstm':
        expected_price_dim = state_dict['price_lstm.weight_ih_l0'].shape[1]
        expected_fund_dim = state_dict['fund_lstm.weight_ih_l0'].shape[1]
        
        # Get price columns (OHLCV for this commodity)
        price_cols = [c for c in groups['price'] if c != target_col] + groups['volume']
        fund_cols = groups['wasde'] + groups['crop_progress'] + groups['weather'] + groups['time']
        
        # Ensure we have the right number of columns
        # If we have more columns than expected, take the first N
        # If we have fewer, we'll need to pad (but this shouldn't happen)
        if len(price_cols) > expected_price_dim:
            price_cols = price_cols[:expected_price_dim]
        if len(fund_cols) > expected_fund_dim:
            fund_cols = fund_cols[:expected_fund_dim]
        
        print(f"  Price columns: {len(price_cols)} (expected {expected_price_dim})")
        print(f"  Fund columns: {len(fund_cols)} (expected {expected_fund_dim})")
        
        # Create datasets manually
        n = len(df)
        train_end = int(n * config['train_split'])
        val_end = int(n * (config['train_split'] + config['val_split']))
        
        test_df = df.iloc[val_end:].copy()
        test_dataset = DualStreamDataset(
            test_df, price_cols, fund_cols, target_col,
            seq_len=config['seq_len'],
            pred_horizon=config['pred_horizon']
        )
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    else:
        # For non-dual-stream models, use standard dataloader
        _, _, test_loader = create_dataloaders(
            df, feature_cols, target_col,
            seq_len=config['seq_len'],
            pred_horizon=config['pred_horizon'],
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            val_split=config['val_split'],
            model_type=config['model_type']
        )
    
    # Recreate model with inferred dimensions
    state_dict = checkpoint['model_state_dict']
    
    if config['model_type'] == 'dual_stream_lstm':
        model = create_model(
            config['model_type'],
            price_input_dim=expected_price_dim,
            fund_input_dim=expected_fund_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=1,
            dropout=config['dropout']
        )
    else:
        # Get input dim from first layer weights
        first_weight_key = [k for k in state_dict.keys() if 'weight' in k][0]
        input_dim = state_dict[first_weight_key].shape[1]
        
        if config['model_type'] == 'resnet':
            model = create_model(
                config['model_type'],
                input_dim=input_dim,
                seq_len=config['seq_len'],
                num_blocks=config.get('num_blocks', 3),
                hidden_dim=config['hidden_dim'],
                output_dim=1,
                dropout=config['dropout']
            )
        elif config['model_type'] == 'transformer':
            model = create_model(
                config['model_type'],
                input_dim=input_dim,
                d_model=config.get('d_model', 128),
                nhead=config.get('nhead', 8),
                num_layers=config['num_layers'],
                dim_feedforward=config.get('dim_feedforward', 256),
                output_dim=1,
                dropout=config['dropout']
            )
        else:  # GRU, single_stream_lstm, lstm_ablation
            model = create_model(
                config['model_type'],
                input_dim=input_dim,
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                output_dim=1,
                dropout=config['dropout']
            )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Generate predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if config['model_type'] == 'dual_stream_lstm':
                # DataLoader returns list: [price_batch, fund_batch, targets]
                # Each is already batched: (batch_size, seq_len, features)
                price_x = batch[0].to(device)
                fund_x = batch[1].to(device)
                targets = batch[2].to(device)
                outputs = model(price_x, fund_x)
                all_preds.extend(outputs.cpu().numpy().flatten().tolist())
                all_targets.extend(targets.cpu().numpy().flatten().tolist())
            else:
                # DataLoader returns list: [x_batch, targets]
                x = batch[0].to(device)
                targets = batch[1].to(device)
                outputs = model(x)
                all_preds.extend(outputs.cpu().numpy().flatten().tolist())
                all_targets.extend(targets.cpu().numpy().flatten().tolist())
    
    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    
    # Load existing results and add predictions
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    results['test_metrics']['predictions'] = predictions.tolist()
    results['test_metrics']['targets'] = targets.tolist()
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Extracted {len(predictions)} predictions")
    print(f"  Test MSE: {results['test_metrics']['mse']:.6f}")
    print(f"  Updated {results_file}")
    
    return results


def extract_all_predictions(results_dir: Path, data_path: Path):
    """Extract predictions for all saved models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find all model files
    model_files = list(results_dir.glob("*_best.pt"))
    
    print(f"Found {len(model_files)} saved models")
    
    success_count = 0
    
    for model_file in sorted(model_files):
        # Parse filename: {model}_{commodity}_h{horizon}_best.pt
        parts = model_file.stem.replace('_best', '').split('_')
        if len(parts) >= 3:
            # Handle model names that may contain underscores
            horizon = int(parts[-1].replace('h', ''))
            commodity = parts[-2]
            model_name = '_'.join(parts[:-2])
            
            try:
                result = extract_predictions(
                    model_name, commodity, horizon,
                    results_dir, data_path, device
                )
                if result:
                    success_count += 1
            except Exception as e:
                print(f"  Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Successfully extracted predictions for {success_count}/{len(model_files)} models")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract predictions from saved models')
    
    parser.add_argument('--model', type=str, default='dual_stream_lstm',
                       help='Model name')
    parser.add_argument('--commodity', type=str, default='corn',
                       choices=['corn', 'soybeans', 'wheat'])
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='results_all_models',
                       help='Directory with saved models and results')
    parser.add_argument('--data', type=str, default='../merged_data/daily_unified.csv')
    parser.add_argument('--all', action='store_true',
                       help='Extract predictions for all saved models')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    data_path = Path(args.data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.all:
        extract_all_predictions(results_dir, data_path)
    else:
        extract_predictions(
            args.model, args.commodity, args.horizon,
            results_dir, data_path, device
        )
