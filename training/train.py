"""
Training Script for Crop Futures Prediction Models

Trains and evaluates 4 architectures:
- Dual-Stream LSTM
- ResNet1D
- Transformer
- GRU

Supports:
- Multi-commodity training (corn, soybeans, wheat)
- Multiple prediction horizons (1-day, 5-day, etc.)
- Hyperparameter tracking
- Model checkpointing
- Evaluation metrics (MSE, MAE, directional accuracy)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import create_model
from dataset import load_unified_data, get_feature_groups, create_dataloaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device, model_type='gru'):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        if model_type == 'dual_stream_lstm':
            price_x, fund_x, y = batch
            price_x = price_x.to(device)
            fund_x = fund_x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(price_x, fund_x)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
        
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, data_loader, criterion, device, model_type='gru'):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            if model_type == 'dual_stream_lstm':
                price_x, fund_x, y = batch
                price_x = price_x.to(device)
                fund_x = fund_x.to(device)
                y = y.to(device)
                outputs = model(price_x, fund_x)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
            
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # Directional accuracy (for financial prediction)
    pred_direction = np.sign(all_preds[1:] - all_preds[:-1])
    true_direction = np.sign(all_targets[1:] - all_targets[:-1])
    directional_acc = np.mean(pred_direction == true_direction)
    
    return {
        'loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'directional_acc': directional_acc,
        'predictions': all_preds,
        'targets': all_targets
    }


def train_model(config, df, feature_cols, target_col, save_dir):
    """
    Train a single model with given configuration.
    
    Args:
        config: Dictionary with training hyperparameters
        df: Unified dataframe
        feature_cols: List of feature column names
        target_col: Target column name
        save_dir: Directory to save model and results
    
    Returns:
        Dictionary with training history and best metrics
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        df, feature_cols, target_col,
        seq_len=config['seq_len'],
        pred_horizon=config['pred_horizon'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        val_split=config['val_split'],
        model_type=config['model_type']
    )
    
    # Determine input dimensions
    if config['model_type'] == 'dual_stream_lstm':
        groups = get_feature_groups(df)
        price_cols = [c for c in groups['price'] if c != target_col]
        price_dim = len(price_cols) + len(groups['volume'])
        fund_dim = len(groups['wasde']) + len(groups['crop_progress']) + len(groups['weather']) + len(groups['time'])
        
        model = create_model(
            config['model_type'],
            price_input_dim=price_dim,
            fund_input_dim=fund_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=1,
            dropout=config['dropout']
        )
    else:
        input_dim = len(feature_cols)
        
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
        else:  # GRU
            model = create_model(
                config['model_type'],
                input_dim=input_dim,
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                output_dim=1,
                dropout=config['dropout'],
                bidirectional=config.get('bidirectional', False)
            )
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']}, Parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.get('early_stopping_patience', 15))
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_dir_acc': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                device, config['model_type'])
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, config['model_type'])
        
        # Update history (convert to float for JSON serialization)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_mse'].append(float(val_metrics['mse']))
        history['val_mae'].append(float(val_metrics['mae']))
        history['val_dir_acc'].append(float(val_metrics['directional_acc']))
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_metrics['loss']:.6f}, MSE: {val_metrics['mse']:.6f}, "
              f"MAE: {val_metrics['mae']:.6f}, Dir Acc: {val_metrics['directional_acc']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }
            
            save_path = save_dir / f"{config['model_name']}_best.pt"
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")
        
        # Early stopping check
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model for final evaluation
    checkpoint = torch.load(save_dir / f"{config['model_name']}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_metrics = evaluate(model, test_loader, criterion, device, config['model_type'])
    
    print(f"\n{'='*50}")
    print(f"Final Test Results - {config['model_name']}")
    print(f"{'='*50}")
    print(f"MSE: {test_metrics['mse']:.6f}")
    print(f"MAE: {test_metrics['mae']:.6f}")
    print(f"Directional Accuracy: {test_metrics['directional_acc']:.4f}")
    print(f"Best Val Epoch: {best_epoch + 1}")
    
    # Save results
    results = {
        'config': config,
        'history': history,
        'best_epoch': best_epoch,
        'test_metrics': {
            'mse': float(test_metrics['mse']),
            'mae': float(test_metrics['mae']),
            'directional_acc': float(test_metrics['directional_acc'])
        },
        'n_parameters': n_params
    }
    
    results_path = save_dir / f"{config['model_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, save_dir, config['model_name'])
    
    # Plot predictions vs targets
    plot_predictions(test_metrics['predictions'], test_metrics['targets'], 
                    save_dir, config['model_name'])
    
    return results


def plot_training_curves(history, save_dir, model_name):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['val_mse'])
    axes[0, 1].set_title('Validation MSE')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['val_mae'])
    axes[1, 0].set_title('Validation MAE')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['val_dir_acc'])
    axes[1, 1].set_title('Validation Directional Accuracy')
    axes[1, 1].grid(True)
    
    plt.suptitle(f'{model_name} Training Curves')
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_curves.png', dpi=150)
    plt.close()


def plot_predictions(predictions, targets, save_dir, model_name):
    """Plot predictions vs actual targets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time series plot
    axes[0].plot(targets, label='Actual', alpha=0.7)
    axes[0].plot(predictions, label='Predicted', alpha=0.7)
    axes[0].set_title(f'{model_name}: Predictions vs Actual')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Normalized Price')
    axes[0].legend()
    axes[0].grid(True)
    
    # Scatter plot
    axes[1].scatter(targets, predictions, alpha=0.5)
    axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
                'r--', label='Perfect Prediction')
    axes[1].set_title(f'{model_name}: Prediction Scatter')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_predictions.png', dpi=150)
    plt.close()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train crop futures prediction models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'dual_stream_lstm', 'resnet', 'transformer', 'gru'],
                       help='Model to train')
    parser.add_argument('--commodity', type=str, default='corn',
                       choices=['corn', 'soybeans', 'wheat'],
                       help='Commodity to predict')
    parser.add_argument('--horizon', type=int, default=1,
                       help='Prediction horizon (days)')
    parser.add_argument('--data', type=str, default='../merged_data/daily_unified.csv',
                       help='Path to unified data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=20,
                       help='Sequence length (lookback window)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for LSTM/GRU/ResNet')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Target column based on commodity
    target_map = {
        'corn': 'cor_close',
        'soybeans': 'soy_close',
        'wheat': 'whe_close'
    }
    target_col = target_map[args.commodity]

    # Load data
    print(f"Loading data from {args.data}")
    df = load_unified_data(args.data)

    # Get feature groups (exclude target column)
    groups = get_feature_groups(df)
    feature_cols = sum(groups.values(), [])
    feature_cols = [c for c in feature_cols if c != target_col]
    
    if target_col not in df.columns:
        print(f"Error: Target column {target_col} not found!")
        print(f"Available columns: {[c for c in df.columns if '_close' in c]}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configurations
    base_config = {
        'commodity': args.commodity,
        'target_col': target_col,
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
    
    # Define models to train
    if args.model == 'all':
        models_to_train = ['dual_stream_lstm', 'resnet', 'transformer', 'gru']
    else:
        models_to_train = [args.model]
    
    # Train each model
    all_results = {}
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        config = base_config.copy()
        config['model_type'] = model_type
        config['model_name'] = f"{model_type}_{args.commodity}_h{args.horizon}"
        
        # Model-specific hyperparameters
        if model_type == 'resnet':
            config['num_blocks'] = 3
        elif model_type == 'transformer':
            config['d_model'] = 128
            config['nhead'] = 8
            config['dim_feedforward'] = 256
        elif model_type == 'gru':
            config['bidirectional'] = False
        
        # Train
        results = train_model(config, df, feature_cols, target_col, output_dir)
        all_results[model_type] = results
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'MSE':<12} {'MAE':<12} {'Dir Acc':<12}")
    print("-" * 60)
    
    for model_type, results in all_results.items():
        test = results['test_metrics']
        print(f"{model_type:<20} {test['mse']:<12.6f} {test['mae']:<12.6f} {test['directional_acc']:<12.4f}")
    
    # Save comparison
    comparison_path = output_dir / 'model_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
