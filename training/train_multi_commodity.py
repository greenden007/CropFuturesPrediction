#!/usr/bin/env python3
"""
Multi-Commodity Training Script

Trains a single model jointly on corn, soybeans, and wheat.
Supports:
- Multi-task learning with shared representations
- Commodity-specific heads
- Cross-commodity attention
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from multi_commodity_models import create_multi_commodity_model
from dataset import load_unified_data, get_feature_groups


class MultiCommodityDataset(Dataset):
    """
    Dataset that combines all three commodities.
    
    Each sample is tagged with its commodity index (0=corn, 1=soybeans, 2=wheat).
    """
    def __init__(self, df, feature_cols, target_cols, seq_len=20, pred_horizon=1):
        """
        Args:
            df: DataFrame with unified data
            feature_cols: List of feature column names
            target_cols: Dict mapping commodity names to target columns
                        e.g., {'corn': 'cor_close', 'soybeans': 'soy_close', ...}
            seq_len: Sequence length
            pred_horizon: Prediction horizon
        """
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        
        # Commodity mapping
        self.commodity_map = {'corn': 0, 'soybeans': 1, 'wheat': 2}
        self.commodity_names = list(target_cols.keys())
        
        # Prepare data for each commodity
        self.samples = []
        
        for commodity_name, target_col in target_cols.items():
            if target_col not in df.columns:
                print(f"Warning: {target_col} not found, skipping {commodity_name}")
                continue
            
            commodity_idx = self.commodity_map[commodity_name]
            
            # Extract valid rows for this commodity
            valid_df = df[feature_cols + [target_col, 'date']].dropna()
            valid_df = valid_df.sort_values('date')
            
            data = valid_df[feature_cols].values.astype(np.float32)
            targets = valid_df[target_col].values.astype(np.float32)
            dates = valid_df['date'].values
            
            # Normalize
            for i in range(data.shape[1]):
                mean = np.nanmean(data[:, i])
                std = np.nanstd(data[:, i])
                if std > 0:
                    data[:, i] = (data[:, i] - mean) / (std + 1e-8)
                data[:, i] = np.nan_to_num(data[:, i], nan=0.0)
            
            # Create sequences
            max_idx = len(data) - seq_len - pred_horizon + 1
            
            for i in range(max_idx):
                seq = data[i:i + seq_len]
                target_idx = i + seq_len + pred_horizon - 1
                target = targets[target_idx]
                
                if not np.isnan(seq).any() and not np.isnan(np.atleast_1d(target)).any():
                    self.samples.append({
                        'sequence': seq,
                        'target': target,
                        'commodity_idx': commodity_idx,
                        'date': dates[target_idx],
                        'commodity_name': commodity_name
                    })
        
        print(f"Created {len(self.samples)} total samples across {len(self.commodity_names)} commodities")
        
        # Print commodity distribution
        counts = {}
        for s in self.samples:
            name = s['commodity_name']
            counts[name] = counts.get(name, 0) + 1
        print("Distribution:", counts)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['sequence']),
            torch.FloatTensor([sample['target']]),
            torch.tensor(sample['commodity_idx'], dtype=torch.long),
            sample['commodity_name']
        )


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        x, y, commodity_idx, _ = batch
        x = x.to(device)
        y = y.to(device)
        commodity_idx = commodity_idx.to(device)
        
        optimizer.zero_grad()
        outputs = model(x, commodity_idx)
        
        loss = criterion(outputs, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, data_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    commodity_losses = {0: [], 1: [], 2: []}
    commodity_names = {0: 'corn', 1: 'soybeans', 2: 'wheat'}
    
    all_preds = {0: [], 1: [], 2: []}
    all_targets = {0: [], 1: [], 2: []}
    
    with torch.no_grad():
        for batch in data_loader:
            x, y, commodity_idx, _ = batch
            x = x.to(device)
            y = y.to(device)
            commodity_idx = commodity_idx.to(device)
            
            outputs = model(x, commodity_idx)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            # Track per-commodity metrics
            for i in range(len(commodity_idx)):
                c_idx = commodity_idx[i].item()
                commodity_losses[c_idx].append(
                    (outputs[i].item() - y[i].item()) ** 2
                )
                all_preds[c_idx].append(outputs[i].item())
                all_targets[c_idx].append(y[i].item())
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate per-commodity metrics
    metrics = {
        'overall': {
            'loss': avg_loss,
            'mse': np.mean([np.mean(commodity_losses[c]) if commodity_losses[c] else 0 
                         for c in [0, 1, 2]]),
        }
    }
    
    for c_idx in [0, 1, 2]:
        if commodity_losses[c_idx]:
            preds = np.array(all_preds[c_idx])
            targets = np.array(all_targets[c_idx])
            
            mse = np.mean((preds - targets) ** 2)
            mae = np.mean(np.abs(preds - targets))
            
            # Directional accuracy
            if len(preds) > 1:
                pred_dir = np.sign(preds[1:] - preds[:-1])
                true_dir = np.sign(targets[1:] - targets[:-1])
                dir_acc = np.mean(pred_dir == true_dir)
            else:
                dir_acc = 0.0
            
            metrics[commodity_names[c_idx]] = {
                'mse': mse,
                'mae': mae,
                'directional_acc': dir_acc,
                'n_samples': len(preds)
            }
        else:
            metrics[commodity_names[c_idx]] = {
                'mse': float('inf'),
                'mae': float('inf'),
                'directional_acc': 0.0,
                'n_samples': 0
            }
    
    return metrics


def train_multi_commodity_model(config, df, feature_cols, target_cols, save_dir):
    """
    Train multi-commodity model.
    
    Args:
        config: Training configuration
        df: Unified dataframe
        feature_cols: Feature columns
        target_cols: Dict of commodity -> target column
        save_dir: Directory to save results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    full_dataset = MultiCommodityDataset(
        df, feature_cols, target_cols,
        seq_len=config['seq_len'],
        pred_horizon=config['pred_horizon']
    )
    
    # Temporal split
    n = len(full_dataset)
    train_end = int(n * config['train_split'])
    val_end = int(n * (config['train_split'] + config['val_split']))
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_end))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(full_dataset, range(val_end, n))
    
    print(f"Split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)
    
    # Create model
    input_dim = len(feature_cols)
    
    if config['model_type'] == 'lstm':
        model = create_multi_commodity_model(
            'lstm',
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_commodities=3,
            dropout=config['dropout']
        )
    elif config['model_type'] == 'transformer':
        model = create_multi_commodity_model(
            'transformer',
            input_dim=input_dim,
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config['num_layers'],
            num_commodities=3,
            dropout=config['dropout']
        )
    else:  # cross_attn
        model = create_multi_commodity_model(
            'cross_attn',
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_commodities=3,
            dropout=config['dropout']
        )
    
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']}, Parameters: {n_params:,}")
    
    # Optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                          weight_decay=config.get('weight_decay', 0.0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    history = {'train_loss': [], 'val_metrics': []}
    best_val_mse = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val - Corn MSE: {val_metrics['corn']['mse']:.6f}, "
              f"Soy MSE: {val_metrics['soybeans']['mse']:.6f}, "
              f"Wheat MSE: {val_metrics['wheat']['mse']:.6f}")
        
        # Check for best model
        avg_val_mse = np.mean([
            val_metrics['corn']['mse'],
            val_metrics['soybeans']['mse'],
            val_metrics['wheat']['mse']
        ])
        
        scheduler.step(avg_val_mse)
        
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, save_dir / f"{config['model_name']}_best.pt")
            print(f"Saved best model (val MSE: {avg_val_mse:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 15):
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Final evaluation
    checkpoint = torch.load(save_dir / f"{config['model_name']}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*50}")
    print("FINAL TEST RESULTS")
    print(f"{'='*50}")
    for commodity in ['corn', 'soybeans', 'wheat']:
        m = test_metrics[commodity]
        print(f"{commodity.upper():<10} MSE: {m['mse']:.6f}, "
              f"MAE: {m['mae']:.6f}, Dir Acc: {m['directional_acc']:.4f}")
    
    # Save results
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'n_parameters': n_params
    }
    
    with open(save_dir / f"{config['model_name']}_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train multi-commodity model')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'cross_attn'],
                       help='Model architecture')
    parser.add_argument('--horizon', type=int, default=1,
                       help='Prediction horizon (days)')
    parser.add_argument('--data', type=str, default='../merged_data/daily_unified.csv',
                       help='Path to unified data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=20,
                       help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--output_dir', type=str, default='./results_multi',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}")
    df = load_unified_data(args.data)
    
    # Define targets for all commodities
    target_cols = {
        'corn': 'cor_close',
        'soybeans': 'soy_close',
        'wheat': 'whe_close'
    }
    
    # Get features (exclude target columns)
    groups = get_feature_groups(df)
    feature_cols = sum(groups.values(), [])
    target_col_values = set(target_cols.values())
    feature_cols = [c for c in feature_cols if c not in target_col_values]
    
    # Verify targets exist
    for name, col in target_cols.items():
        if col not in df.columns:
            print(f"Warning: {col} not found in data")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = {
        'model_type': args.model,
        'model_name': f"multi_{args.model}_h{args.horizon}",
        'seq_len': args.seq_len,
        'pred_horizon': args.horizon,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.epochs,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'train_split': 0.7,
        'val_split': 0.15,
        'early_stopping_patience': 15,
        'weight_decay': 1e-5
    }
    
    if args.model == 'transformer':
        config['d_model'] = 128
        config['nhead'] = 8
        config['dim_feedforward'] = 256
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training Multi-Commodity {args.model.upper()}")
    print(f"{'='*60}")
    
    results = train_multi_commodity_model(config, df, feature_cols, target_cols, output_dir)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
