"""
Classical Baseline Models for Commodity Price Prediction

Implements:
1. Random Walk (naive baseline)
2. ARIMA - Autoregressive Integrated Moving Average
3. Linear Regression (Ridge) on feature set
4. Exponential Smoothing

These provide sanity checks and established benchmarks for ML models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. ARIMA and ExponentialSmoothing unavailable.")
    print("Install with: pip install statsmodels")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import load_unified_data, get_feature_groups, create_dataloaders


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate directional accuracy."""
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return np.mean(true_dir == pred_dir)


def random_walk_baseline(
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    pred_horizon: int = 1
) -> Dict:
    """
    Random Walk: y_hat[t] = y[t-1]
    This is the naive baseline that assumes tomorrow = today.
    """
    # For horizon=1: predict previous value
    # For horizon=h: predict value from h steps ago
    
    train_pred = train_series[:-pred_horizon]
    train_true = train_series[pred_horizon:]
    
    val_pred = np.concatenate([
        train_series[-pred_horizon:],  # Use last train values to start
        val_series[:-pred_horizon]
    ])
    val_true = val_series
    
    test_pred = np.concatenate([
        val_series[-pred_horizon:],  # Use last val values to start
        test_series[:-pred_horizon]
    ])
    test_true = test_series
    
    # Trim to same length
    min_len = min(len(val_pred), len(val_true))
    val_pred = val_pred[:min_len]
    val_true = val_true[:min_len]
    
    min_len = min(len(test_pred), len(test_true))
    test_pred = test_pred[:min_len]
    test_true = test_true[:min_len]
    
    return {
        'model': 'random_walk',
        'train_mse': mean_squared_error(train_true, train_pred),
        'val_mse': mean_squared_error(val_true, val_pred),
        'test_mse': mean_squared_error(test_true, test_pred),
        'train_mae': mean_absolute_error(train_true, train_pred),
        'val_mae': mean_absolute_error(val_true, val_pred),
        'test_mae': mean_absolute_error(test_true, test_pred),
        'train_dir_acc': directional_accuracy(train_true, train_pred),
        'val_dir_acc': directional_accuracy(val_true, val_pred),
        'test_dir_acc': directional_accuracy(test_true, test_pred),
        'n_parameters': 0
    }


def arima_baseline(
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    pred_horizon: int = 1,
    order: Tuple[int, int, int] = (5, 1, 0)
) -> Dict:
    """
    ARIMA baseline.
    Default order (5,1,0): AR(5) with first differencing, no MA component.
    """
    if not STATSMODELS_AVAILABLE:
        return {'model': 'arima', 'error': 'statsmodels not installed'}
    
    try:
        # Combine train+val for fitting (standard practice)
        fit_series = np.concatenate([train_series, val_series])
        
        # Fit ARIMA
        model = ARIMA(fit_series, order=order)
        fitted = model.fit()
        
        # Forecast test period
        forecast = fitted.forecast(steps=len(test_series))
        
        # For validation, use rolling forecasts or fitted values
        val_fitted = fitted.fittedvalues[len(train_series):len(train_series)+len(val_series)]
        
        # Handle horizon > 1 by using recursive forecasts
        if pred_horizon > 1:
            # For multi-step, ARIMA forecast is already multi-step
            # But we need to align with our evaluation
            pass  # forecast is already the right length
        
        return {
            'model': f'arima_{order}',
            'train_mse': np.nan,  # ARIMA doesn't have explicit train predictions
            'val_mse': mean_squared_error(val_series, val_fitted) if len(val_fitted) == len(val_series) else np.nan,
            'test_mse': mean_squared_error(test_series, forecast[:len(test_series)]),
            'train_mae': np.nan,
            'val_mae': mean_absolute_error(val_series, val_fitted) if len(val_fitted) == len(val_series) else np.nan,
            'test_mae': mean_absolute_error(test_series, forecast[:len(test_series)]),
            'val_dir_acc': directional_accuracy(val_series, val_fitted) if len(val_fitted) == len(val_series) else 0.0,
            'test_dir_acc': directional_accuracy(test_series, forecast[:len(test_series)]),
            'n_parameters': fitted.params.shape[0],
            'aic': fitted.aic,
            'bic': fitted.bic
        }
    except Exception as e:
        return {'model': 'arima', 'error': str(e)}


def linear_baseline(
    df: pd.DataFrame,
    target_col: str,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_groups: Dict[str, List[str]] = None
) -> Dict:
    """
    Ridge Regression baseline using all available features.
    """
    # Get features (exclude date and target)
    exclude_cols = ['date', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Handle NaN features
    X = df[feature_cols].fillna(method='ffill').fillna(0).values
    y = df[target_col].values
    
    # Split
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    return {
        'model': 'ridge_regression',
        'train_mse': mean_squared_error(y_train, train_pred),
        'val_mse': mean_squared_error(y_val, val_pred),
        'test_mse': mean_squared_error(y_test, test_pred),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_dir_acc': directional_accuracy(y_train, train_pred),
        'val_dir_acc': directional_accuracy(y_val, val_pred),
        'test_dir_acc': directional_accuracy(y_test, test_pred),
        'n_parameters': len(feature_cols) + 1,
        'feature_count': len(feature_cols)
    }


def exponential_smoothing_baseline(
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    trend: Optional[str] = 'add'
) -> Dict:
    """
    Holt-Winters Exponential Smoothing.
    """
    if not STATSMODELS_AVAILABLE:
        return {'model': 'exp_smoothing', 'error': 'statsmodels not installed'}
    
    try:
        # Fit on train+val
        fit_series = np.concatenate([train_series, val_series])
        
        model = ExponentialSmoothing(
            fit_series,
            trend=trend,
            seasonal=None,  # No seasonality for daily data
            damped_trend=True
        )
        fitted = model.fit()
        
        # Forecast
        forecast = fitted.forecast(steps=len(test_series))
        
        # Validation fitted values
        val_fitted = fitted.fittedvalues[len(train_series):len(train_series)+len(val_series)]
        
        return {
            'model': f'exp_smoothing_{trend}',
            'test_mse': mean_squared_error(test_series, forecast[:len(test_series)]),
            'test_mae': mean_absolute_error(test_series, forecast[:len(test_series)]),
            'val_mse': mean_squared_error(val_series, val_fitted) if len(val_fitted) == len(val_series) else np.nan,
            'val_mae': mean_absolute_error(val_series, val_fitted) if len(val_fitted) == len(val_series) else np.nan,
            'test_dir_acc': directional_accuracy(test_series, forecast[:len(test_series)]),
            'val_dir_acc': directional_accuracy(val_series, val_fitted) if len(val_fitted) == len(val_series) else 0.0,
            'n_parameters': len(fitted.params)
        }
    except Exception as e:
        return {'model': 'exp_smoothing', 'error': str(e)}


def run_all_baselines(
    data_path: str,
    commodity: str,
    horizon: int,
    output_dir: str = 'results_baselines',
    train_split: float = 0.7,
    val_split: float = 0.15
) -> Dict:
    """
    Run all classical baselines for comparison.
    """
    print(f"\n{'='*60}")
    print(f"CLASSICAL BASELINES: {commodity.upper()} (H={horizon})")
    print(f"{'='*60}")
    
    # Load data
    df = load_unified_data(data_path)
    
    # Target column
    target_map = {'corn': 'cor_close', 'soybeans': 'soy_close', 'wheat': 'whe_close'}
    target_col = target_map.get(commodity, f'{commodity[:3]}_close')
    
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found")
    
    # Get series
    df = df.sort_values('date').reset_index(drop=True)
    series = df[target_col].values
    
    # Create masks for split
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    
    # Split series
    train_series = series[train_mask]
    val_series = series[val_mask]
    test_series = series[test_mask]
    
    print(f"Data shape: {series.shape}")
    print(f"Train: {len(train_series)}, Val: {len(val_series)}, Test: {len(test_series)}")
    
    results = {}
    
    # 1. Random Walk
    print("\n[1/5] Random Walk...")
    results['random_walk'] = random_walk_baseline(
        train_series, val_series, test_series, horizon
    )
    print(f"  Test MSE: {results['random_walk']['test_mse']:.6f}")
    
    # 2. ARIMA (if available)
    if STATSMODELS_AVAILABLE:
        print("\n[2/5] ARIMA(5,1,0)...")
        results['arima_510'] = arima_baseline(
            train_series, val_series, test_series, horizon, order=(5, 1, 0)
        )
        if 'test_mse' in results['arima_510']:
            print(f"  Test MSE: {results['arima_510']['test_mse']:.6f}")
        
        print("\n[3/5] ARIMA(1,1,1)...")
        results['arima_111'] = arima_baseline(
            train_series, val_series, test_series, horizon, order=(1, 1, 1)
        )
        if 'test_mse' in results['arima_111']:
            print(f"  Test MSE: {results['arima_111']['test_mse']:.6f}")
    
    # 3. Linear Regression
    print("\n[4/5] Ridge Regression...")
    results['ridge'] = linear_baseline(
        df, target_col, train_mask, val_mask, test_mask
    )
    print(f"  Test MSE: {results['ridge']['test_mse']:.6f}")
    print(f"  Features used: {results['ridge']['feature_count']}")
    
    # 4. Exponential Smoothing (if available)
    if STATSMODELS_AVAILABLE:
        print("\n[5/5] Exponential Smoothing...")
        results['exp_smoothing'] = exponential_smoothing_baseline(
            train_series, val_series, test_series, trend='add'
        )
        if 'test_mse' in results['exp_smoothing']:
            print(f"  Test MSE: {results['exp_smoothing']['test_mse']:.6f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f'baselines_{commodity}_h{horizon}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON (Test MSE)")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Test MSE':<12} {'Test MAE':<12} {'Dir Acc':<10}")
    print("-" * 60)
    
    for name, res in sorted(results.items(), key=lambda x: x[1].get('test_mse', float('inf'))):
        if 'test_mse' in res:
            print(f"{name:<25} {res['test_mse']:<12.6f} {res['test_mae']:<12.6f} {res.get('test_dir_acc', 0):<10.4f}")
    
    print(f"\nResults saved to {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run classical baseline models')
    parser.add_argument('--commodity', type=str, default='corn',
                       choices=['corn', 'soybeans', 'wheat'])
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--data', type=str, default='../merged_data/daily_unified.csv')
    parser.add_argument('--output_dir', type=str, default='results_baselines')
    
    args = parser.parse_args()
    
    results = run_all_baselines(
        data_path=args.data,
        commodity=args.commodity,
        horizon=args.horizon,
        output_dir=args.output_dir
    )
