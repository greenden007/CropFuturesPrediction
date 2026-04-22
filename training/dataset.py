"""
Dataset and DataLoader for Crop Futures Prediction

Handles loading unified datasets and creating sequences for time series forecasting.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FuturesDataset(Dataset):
    """
    PyTorch Dataset for crop futures time series data.
    
    Creates sequences of historical data for predicting future price movements.
    
    Args:
        df: DataFrame with unified data
        feature_cols: List of feature column names
        target_col: Target column name (e.g., 'cor_close' for corn close price)
        seq_len: Length of input sequence (lookback window)
        pred_horizon: Prediction horizon (1=next day, 5=next week, etc.)
        normalize: Whether to normalize features
        price_cols: Columns that are prices (normalized separately)
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], 
                 target_col: str, seq_len: int = 20, pred_horizon: int = 1,
                 normalize: bool = True, price_cols: Optional[List[str]] = None):
        
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.target_col = target_col
        self.normalize = normalize
        
        # Ensure date is sorted
        df = df.sort_values('date').reset_index(drop=True)
        
        # Store dates for reference
        self.dates = df['date'].values
        
        # Select features
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        self.data = df[self.feature_cols].values.astype(np.float32)
        
        # Target
        if target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            raise ValueError(f"Target column {target_col} not found in dataframe")
        
        # Price columns for special normalization
        self.price_cols = price_cols or []
        
        # Normalization
        if normalize:
            self._normalize()
        
        # Create sequences
        self._create_sequences()
    
    def _normalize(self):
        """Z-score normalization with price-specific handling."""
        # For price columns: normalize by recent history
        price_indices = [self.feature_cols.index(c) for c in self.price_cols 
                        if c in self.feature_cols]
        
        for i in range(self.data.shape[1]):
            col_data = self.data[:, i]
            
            if i in price_indices:
                # For prices: use rolling window normalization
                # But we need to do this carefully to avoid look-ahead bias
                # For now, use global mean/std for simplicity
                # In production, use expanding window normalization
                mean = np.nanmean(col_data)
                std = np.nanstd(col_data)
            else:
                # For other features: global normalization
                mean = np.nanmean(col_data)
                std = np.nanstd(col_data)
            
            if std > 0:
                self.data[:, i] = (col_data - mean) / (std + 1e-8)
            
            # Fill any remaining NaNs with 0
            self.data[:, i] = np.nan_to_num(self.data[:, i], nan=0.0)
        
        # Also normalize target
        self.target_mean = np.nanmean(self.targets)
        self.target_std = np.nanstd(self.targets)
        if self.target_std > 0:
            self.targets = (self.targets - self.target_mean) / (self.target_std + 1e-8)
    
    def _create_sequences(self):
        """Create input sequences and targets."""
        self.sequences = []
        self.sequence_targets = []
        
        # We need seq_len history + pred_horizon for target
        max_idx = len(self.data) - self.seq_len - self.pred_horizon + 1
        
        for i in range(max_idx):
            # Input sequence: [i, i+seq_len)
            seq = self.data[i:i + self.seq_len]
            
            # Target: value at i+seq_len+pred_horizon-1
            target_idx = i + self.seq_len + self.pred_horizon - 1
            target = self.targets[target_idx]
            
            # Check for NaN in sequence or target
            if not np.isnan(seq).any() and not np.isnan(target):
                self.sequences.append(seq)
                self.sequence_targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.sequence_targets[idx]])
        )
    
    def get_date(self, idx):
        """Get the date corresponding to a sequence index."""
        # The target date for this sequence
        return self.dates[idx + self.seq_len + self.pred_horizon - 1]


class DualStreamDataset(Dataset):
    """
    Dataset for Dual-Stream LSTM with separate price and fundamental streams.
    
    Args:
        df: DataFrame with unified data
        price_cols: Columns for price stream (OHLCV)
        fund_cols: Columns for fundamentals stream (WASDE, crop progress)
        target_col: Target column name
        seq_len: Sequence length
        pred_horizon: Prediction horizon
    """
    def __init__(self, df: pd.DataFrame, price_cols: List[str], 
                 fund_cols: List[str], target_col: str, seq_len: int = 20,
                 pred_horizon: int = 1):
        
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        
        # Ensure sorted
        df = df.sort_values('date').reset_index(drop=True)
        self.dates = df['date'].values
        
        # Separate streams
        self.price_cols = [c for c in price_cols if c in df.columns]
        self.fund_cols = [c for c in fund_cols if c in df.columns]
        
        self.price_data = df[self.price_cols].values.astype(np.float32)
        self.fund_data = df[self.fund_cols].values.astype(np.float32)
        
        if target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            raise ValueError(f"Target column {target_col} not found")
        
        # Normalize each stream
        self._normalize()
        
        # Create sequences
        self._create_sequences()
    
    def _normalize(self):
        """Normalize each stream separately."""
        # Price stream normalization
        for i in range(self.price_data.shape[1]):
            mean = np.nanmean(self.price_data[:, i])
            std = np.nanstd(self.price_data[:, i])
            if std > 0:
                self.price_data[:, i] = (self.price_data[:, i] - mean) / (std + 1e-8)
            self.price_data[:, i] = np.nan_to_num(self.price_data[:, i], nan=0.0)
        
        # Fundamentals stream normalization
        for i in range(self.fund_data.shape[1]):
            mean = np.nanmean(self.fund_data[:, i])
            std = np.nanstd(self.fund_data[:, i])
            if std > 0:
                self.fund_data[:, i] = (self.fund_data[:, i] - mean) / (std + 1e-8)
            self.fund_data[:, i] = np.nan_to_num(self.fund_data[:, i], nan=0.0)
        
        # Target normalization
        self.target_mean = np.nanmean(self.targets)
        self.target_std = np.nanstd(self.targets)
        if self.target_std > 0:
            self.targets = (self.targets - self.target_mean) / (self.target_std + 1e-8)
    
    def _create_sequences(self):
        """Create sequences for both streams."""
        self.price_sequences = []
        self.fund_sequences = []
        self.sequence_targets = []
        
        max_idx = len(self.price_data) - self.seq_len - self.pred_horizon + 1
        
        for i in range(max_idx):
            price_seq = self.price_data[i:i + self.seq_len]
            fund_seq = self.fund_data[i:i + self.seq_len]
            
            target_idx = i + self.seq_len + self.pred_horizon - 1
            target = self.targets[target_idx]
            
            if (not np.isnan(price_seq).any() and 
                not np.isnan(fund_seq).any() and 
                not np.isnan(target)):
                
                self.price_sequences.append(price_seq)
                self.fund_sequences.append(fund_seq)
                self.sequence_targets.append(target)
    
    def __len__(self):
        return len(self.price_sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.price_sequences[idx]),
            torch.FloatTensor(self.fund_sequences[idx]),
            torch.FloatTensor([self.sequence_targets[idx]])
        )


def load_unified_data(data_path: str = 'merged_data/daily_unified.csv',
                      freq: str = 'daily') -> pd.DataFrame:
    """
    Load unified dataset and prepare for modeling.
    
    Args:
        data_path: Path to unified CSV
        freq: 'daily' or 'weekly'
    
    Returns:
        DataFrame with parsed dates
    """
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Remove rows with all NaN features
    feature_cols = [c for c in df.columns if c != 'date']
    df = df.dropna(subset=feature_cols, how='all')
    
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Automatically group features by type.

    Returns:
        Dictionary with feature group lists
    """
    groups = {
        'price': [],
        'volume': [],
        'wasde': [],
        'crop_progress': [],
        'weather': [],
        'image': [],
        'time': []
    }

    for col in df.columns:
        if col == 'date':
            continue
        elif any(x in col for x in ['_close', '_high', '_low', '_open']):
            groups['price'].append(col)
        elif '_volume' in col:
            groups['volume'].append(col)
        elif 'wasde_' in col:
            groups['wasde'].append(col)
        elif any(x in col for x in ['corn_', 'soybeans_', 'wheat_']):
            groups['crop_progress'].append(col)
        elif 'outlook_' in col or 'has_' in col:
            groups['weather'].append(col)
        elif col.startswith('img_feat_'):
            groups['image'].append(col)
        elif col in ['year', 'month', 'day_of_year', 'week_of_year', 'is_month_end']:
            groups['time'].append(col)

    return groups


def create_dataloaders(df: pd.DataFrame, 
                       feature_cols: List[str],
                       target_col: str,
                       seq_len: int = 20,
                       pred_horizon: int = 1,
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       model_type: str = 'gru') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with temporal split.
    
    Args:
        df: Unified dataframe
        feature_cols: Feature columns to use
        target_col: Target column
        seq_len: Sequence length
        pred_horizon: Prediction horizon
        batch_size: Batch size
        train_split: Fraction for training
        val_split: Fraction for validation (remainder goes to test)
        model_type: 'dual_stream_lstm' or others (determines dataset type)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Temporal split (important for time series!)
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create datasets based on model type
    if model_type == 'dual_stream_lstm':
        # Separate price and fundamental features
        groups = get_feature_groups(df)
        price_cols = [c for c in groups['price'] if c != target_col] + groups['volume']
        fund_cols = groups['wasde'] + groups['crop_progress'] + groups['weather'] + groups['time']
        
        train_dataset = DualStreamDataset(train_df, price_cols, fund_cols, 
                                         target_col, seq_len, pred_horizon)
        val_dataset = DualStreamDataset(val_df, price_cols, fund_cols,
                                       target_col, seq_len, pred_horizon)
        test_dataset = DualStreamDataset(test_df, price_cols, fund_cols,
                                        target_col, seq_len, pred_horizon)
    else:
        # Single stream dataset
        train_dataset = FuturesDataset(train_df, feature_cols, target_col,
                                      seq_len, pred_horizon)
        val_dataset = FuturesDataset(val_df, feature_cols, target_col,
                                    seq_len, pred_horizon)
        test_dataset = FuturesDataset(test_df, feature_cols, target_col,
                                     seq_len, pred_horizon)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    # Load data
    df = load_unified_data('../merged_data/daily_unified.csv')
    
    # Get feature groups
    groups = get_feature_groups(df)
    print("\nFeature groups:")
    for group, cols in groups.items():
        print(f"  {group}: {len(cols)} features")
    
    # Create single stream dataset
    all_features = sum(groups.values(), [])
    target = 'cor_close'  # Predict corn close price
    
    if target in df.columns:
        dataset = FuturesDataset(df, all_features, target, seq_len=20)
        print(f"\nDataset size: {len(dataset)}")
        
        x, y = dataset[0]
        print(f"Sample shape: x={x.shape}, y={y.shape}")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            df, all_features, target, seq_len=20, model_type='gru'
        )
        
        print(f"\nDataloader sizes: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        
        # Test batch
        batch_x, batch_y = next(iter(train_loader))
        print(f"Batch shape: x={batch_x.shape}, y={batch_y.shape}")
    else:
        print(f"\nTarget {target} not found. Available targets: {groups['price']}")
    
    print("\nTests completed!")
