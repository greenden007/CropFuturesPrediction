"""
Multi-Seed Experiment with Statistical Testing

Runs models with multiple random seeds and performs statistical significance tests
to validate that performance differences are real, not due to random initialization.

Statistical Tests:
1. Paired t-test: Tests if mean difference between two models is significant
2. Diebold-Mariano test: Tests if forecast accuracy difference is significant
3. Wilcoxon signed-rank: Non-parametric alternative to t-test

Usage:
    python multi_seed_experiment.py --model1 dual_stream_lstm --model2 gru --n_seeds 5
    python multi_seed_experiment.py --model1 dual_stream_lstm --model2 single_stream_lstm --commodity corn
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple


def run_training_with_seed(
    model: str,
    commodity: str,
    horizon: int,
    seed: int,
    epochs: int = 100,
    output_dir: str = 'results_multi_seed',
    data_path: str = 'merged_data/daily_unified.csv'
) -> Dict:
    """Run training with a specific random seed."""
    
    cmd = [
        'python', 'training/train.py',
        '--model', model,
        '--commodity', commodity,
        '--horizon', str(horizon),
        '--epochs', str(epochs),
        '--output_dir', output_dir,
        '--data', data_path,
        '--seed', str(seed)
    ]
    
    # Set random seed for reproducibility
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        env=env
    )
    
    # Parse results
    model_name = f"{model}_{commodity}_h{horizon}"
    results_file = Path(output_dir) / f"{model_name}_results.json"
    
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            saved_results = json.load(f)
        
        return {
            'seed': seed,
            'success': True,
            'test_mse': saved_results['test_metrics']['mse'],
            'test_mae': saved_results['test_metrics']['mae'],
            'test_dir_acc': saved_results['test_metrics']['directional_acc'],
            'val_mse': saved_results['history']['val_mse'][-1] if saved_results['history']['val_mse'] else None,
            'best_epoch': saved_results.get('best_epoch', 0)
        }
    else:
        return {
            'seed': seed,
            'success': False,
            'error': result.stderr[-500:] if result.stderr else 'Unknown error'
        }


def diebold_mariano_test(
    errors1: np.ndarray, 
    errors2: np.ndarray,
    loss_type: str = 'squared'
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests H0: E[loss(e1)] = E[loss(e2)]
    Returns: (DM statistic, p-value)
    """
    if loss_type == 'squared':
        d = errors1**2 - errors2**2
    elif loss_type == 'absolute':
        d = np.abs(errors1) - np.abs(errors2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    
    if var_d == 0:
        return 0.0, 1.0
    
    # DM statistic (asymptotically standard normal)
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value


def compare_models_statistical(
    model1_results: List[Dict],
    model2_results: List[Dict],
    model1_name: str,
    model2_name: str,
    alpha: float = 0.05
) -> Dict:
    """
    Perform comprehensive statistical comparison between two models.
    """
    # Extract metrics
    mse1 = [r['test_mse'] for r in model1_results if r['success']]
    mse2 = [r['test_mse'] for r in model2_results if r['success']]
    
    mae1 = [r['test_mae'] for r in model1_results if r['success']]
    mae2 = [r['test_mae'] for r in model2_results if r['success']]
    
    dir_acc1 = [r['test_dir_acc'] for r in model1_results if r['success']]
    dir_acc2 = [r['test_dir_acc'] for r in model2_results if r['success']]
    
    results = {
        'model1': model1_name,
        'model2': model2_name,
        'n_seeds': len(mse1),
        'model1_stats': {},
        'model2_stats': {},
        'comparisons': {}
    }
    
    # Descriptive statistics
    for metric_name, vals1, vals2 in [
        ('mse', mse1, mse2),
        ('mae', mae1, mae2),
        ('directional_acc', dir_acc1, dir_acc2)
    ]:
        results['model1_stats'][metric_name] = {
            'mean': float(np.mean(vals1)),
            'std': float(np.std(vals1, ddof=1)),
            'min': float(np.min(vals1)),
            'max': float(np.max(vals1)),
            'median': float(np.median(vals1))
        }
        
        results['model2_stats'][metric_name] = {
            'mean': float(np.mean(vals2)),
            'std': float(np.std(vals2, ddof=1)),
            'min': float(np.min(vals2)),
            'max': float(np.max(vals2)),
            'median': float(np.median(vals2))
        }
        
        # Paired t-test (if same seeds)
        if len(vals1) == len(vals2):
            t_stat, p_value = stats.ttest_rel(vals1, vals2)
            results['comparisons'][f'{metric_name}_paired_ttest'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'winner': model1_name if np.mean(vals1) < np.mean(vals2) else model2_name
            }
        
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(vals1, vals2, equal_var=False)
        results['comparisons'][f'{metric_name}_independent_ttest'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'winner': model1_name if np.mean(vals1) < np.mean(vals2) else model2_name
        }
        
        # Wilcoxon signed-rank test (non-parametric)
        if len(vals1) == len(vals2):
            try:
                w_stat, p_value = stats.wilcoxon(vals1, vals2)
                results['comparisons'][f'{metric_name}_wilcoxon'] = {
                    'w_statistic': float(w_stat),
                    'p_value': float(p_value),
                    'significant': p_value < alpha,
                    'winner': model1_name if np.median(vals1) < np.median(vals2) else model2_name
                }
            except ValueError:
                pass  # If all values are identical
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(vals1, ddof=1)**2 + np.std(vals2, ddof=1)**2) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std
            results['comparisons'][f'{metric_name}_cohens_d'] = {
                'd': float(cohens_d),
                'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
            }
    
    return results


def run_multi_seed_experiment(
    model1: str,
    model2: str,
    commodity: str,
    horizon: int,
    n_seeds: int,
    epochs: int,
    output_dir: str,
    alpha: float = 0.05
) -> Dict:
    """
    Run full multi-seed comparison experiment.
    """
    print(f"\n{'='*70}")
    print(f"MULTI-SEED EXPERIMENT: {model1} vs {model2}")
    print(f"Commodity: {commodity}, Horizon: {horizon}, Seeds: {n_seeds}")
    print(f"{'='*70}\n")
    
    # Generate seeds
    np.random.seed(42)  # For reproducibility of seed generation
    seeds = np.random.randint(0, 10000, size=n_seeds).tolist()
    
    print(f"Using seeds: {seeds}\n")
    
    # Run experiments
    model1_results = []
    model2_results = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[Seed {i}/{n_seeds}] {model1}")
        result = run_training_with_seed(
            model1, commodity, horizon, seed, epochs, output_dir
        )
        model1_results.append(result)
        
        if result['success']:
            print(f"  MSE: {result['test_mse']:.6f}")
        else:
            print(f"  FAILED: {result.get('error', 'Unknown')[:100]}")
        
        print(f"\n[Seed {i}/{n_seeds}] {model2}")
        result = run_training_with_seed(
            model2, commodity, horizon, seed, epochs, output_dir
        )
        model2_results.append(result)
        
        if result['success']:
            print(f"  MSE: {result['test_mse']:.6f}")
        else:
            print(f"  FAILED: {result.get('error', 'Unknown')[:100]}")
    
    # Statistical comparison
    stats_results = compare_models_statistical(
        model1_results, model2_results, model1, model2, alpha
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for metric in ['mse', 'mae', 'directional_acc']:
        print(f"\n{metric.upper()}:")
        m1_stats = stats_results['model1_stats'][metric]
        m2_stats = stats_results['model2_stats'][metric]
        
        print(f"  {model1}: {m1_stats['mean']:.6f} ± {m1_stats['std']:.6f}")
        print(f"  {model2}: {m2_stats['mean']:.6f} ± {m2_stats['std']:.6f}")
        
        # Check if significant
        ttest_key = f'{metric}_paired_ttest'
        if ttest_key in stats_results['comparisons']:
            comp = stats_results['comparisons'][ttest_key]
            sig_marker = "***" if comp['significant'] else "ns"
            print(f"  Paired t-test: p={comp['p_value']:.4f} {sig_marker}")
            print(f"  Winner: {comp['winner']}")
        
        # Effect size
        d_key = f'{metric}_cohens_d'
        if d_key in stats_results['comparisons']:
            d_result = stats_results['comparisons'][d_key]
            print(f"  Cohen's d: {d_result['d']:.3f} ({d_result['effect_size']} effect)")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f'statistical_comparison_{model1}_vs_{model2}_{commodity}_h{horizon}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'model1': model1,
                'model2': model2,
                'commodity': commodity,
                'horizon': horizon,
                'n_seeds': n_seeds,
                'seeds': seeds,
                'epochs': epochs,
                'alpha': alpha
            },
            'model1_results': model1_results,
            'model2_results': model2_results,
            'statistical_tests': stats_results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return stats_results


def run_ablation_study(
    base_model: str,
    commodity: str,
    horizon: int,
    n_seeds: int,
    output_dir: str = 'results_ablation'
) -> Dict:
    """
    Run ablation study comparing full model against variants with removed features.
    """
    ablations = {
        'no_weather': 'Exclude weather outlook features',
        'no_wasde': 'Exclude WASDE features',
        'no_crop_progress': 'Exclude crop progress features',
        'no_image': 'Exclude image features',
        'price_only': 'Price/volume features only',
    }
    
    results = {}
    
    for ablation_name, description in ablations.items():
        print(f"\n{'='*60}")
        print(f"ABLATION: {ablation_name}")
        print(f"Description: {description}")
        print(f"{'='*60}")
        
        # This would need support in train.py for feature filtering
        # For now, placeholder
        print("Note: Requires train.py support for feature filtering")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multi-seed experiment with statistical testing'
    )
    
    parser.add_argument('--model1', type=str, default='dual_stream_lstm',
                       help='First model to compare')
    parser.add_argument('--model2', type=str, default='gru',
                       help='Second model to compare')
    parser.add_argument('--commodity', type=str, default='corn',
                       choices=['corn', 'soybeans', 'wheat'])
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of random seeds to test')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs per run (use fewer for multi-seed)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for statistical tests')
    parser.add_argument('--output_dir', type=str, default='results_multi_seed')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study instead of model comparison')
    
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation_study(
            base_model=args.model1,
            commodity=args.commodity,
            horizon=args.horizon,
            n_seeds=args.n_seeds,
            output_dir=args.output_dir
        )
    else:
        run_multi_seed_experiment(
            model1=args.model1,
            model2=args.model2,
            commodity=args.commodity,
            horizon=args.horizon,
            n_seeds=args.n_seeds,
            epochs=args.epochs,
            output_dir=args.output_dir,
            alpha=args.alpha
        )
