"""
Extract best configuration from wandb sweep and save for final training.

Usage:
    python3 training/get_best_sweep_config.py --sweep_id crop-futures/abc123
    
    # Save top 5 configs
    python3 training/get_best_sweep_config.py --sweep_id crop-futures/abc123 --top_k 5
    
    # Export for specific model
    python3 training/get_best_sweep_config.py --sweep_id crop-futures/abc123 --model attention_vision_dual_stream
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

try:
    import wandb
    import pandas as pd
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Error: wandb and pandas required. Install with: pip install wandb pandas")
    exit(1)


def get_sweep_results(sweep_id: str, entity: str = None, project: str = "crop-futures-prediction") -> List[Dict]:
    """Get all runs from a sweep with their metrics and configs."""
    api = wandb.Api()
    
    # Build full sweep path
    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        # Try to parse from sweep_id
        if "/" in sweep_id:
            parts = sweep_id.split("/")
            if len(parts) == 3:
                sweep_path = sweep_id
            elif len(parts) == 2:
                sweep_path = f"{parts[0]}/{project}/{parts[1]}"
        else:
            sweep_path = f"{project}/{sweep_id}"
    
    print(f"Fetching sweep: {sweep_path}")
    
    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"Error accessing sweep: {e}")
        print("Make sure you're logged in: wandb login")
        return []
    
    # Collect runs
    runs = []
    for run in sweep.runs:
        summary = run.summary
        config = run.config
        
        runs.append({
            'run_id': run.id,
            'state': run.state,
            'config': config,
            'val_directional_accuracy': summary.get('val_directional_accuracy', 0),
            'final_test_directional_accuracy': summary.get('final_test_directional_accuracy', 0),
            'val_mse': summary.get('val_mse', float('inf')),
            'final_test_mse': summary.get('final_test_mse', float('inf')),
            'best_epoch': summary.get('best_epoch', 0),
        })
    
    return runs


def filter_and_sort_runs(runs: List[Dict], model_type: str = None, min_val_acc: float = 0.0) -> List[Dict]:
    """Filter and sort runs by performance."""
    # Filter completed runs
    runs = [r for r in runs if r['state'] == 'finished']
    
    # Filter by model type if specified
    if model_type:
        runs = [r for r in runs if r['config'].get('model') == model_type]
    
    # Filter by minimum validation accuracy
    runs = [r for r in runs if r['val_directional_accuracy'] >= min_val_acc]
    
    # Sort by validation directional accuracy (descending)
    runs.sort(key=lambda x: x['val_directional_accuracy'], reverse=True)
    
    return runs


def export_best_configs(runs: List[Dict], output_dir: Path, top_k: int = 5):
    """Export top K configurations to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting top {min(top_k, len(runs))} configurations...")
    
    for i, run in enumerate(runs[:top_k], 1):
        config = run['config']
        
        # Build filename
        model = config.get('model', 'unknown')
        commodity = config.get('commodity', 'unknown')
        horizon = config.get('horizon', 0)
        
        filename = f"best_config_{i}_{model}_{commodity}_h{horizon}.json"
        filepath = output_dir / filename
        
        # Prepare export
        export = {
            'rank': i,
            'run_id': run['run_id'],
            'val_directional_accuracy': float(run['val_directional_accuracy']),
            'final_test_directional_accuracy': float(run.get('final_test_directional_accuracy', 0)),
            'val_mse': float(run['val_mse']),
            'best_epoch': int(run['best_epoch']),
            'config': config
        }
        
        with open(filepath, 'w') as f:
            json.dump(export, f, indent=2)
        
        print(f"  {i}. {filepath}")
        print(f"     Val Dir Acc: {run['val_directional_accuracy']:.4f}")
        print(f"     Hidden: {config.get('hidden_dim')}, Dropout: {config.get('dropout'):.2f}, LR: {config.get('lr')}")
    
    # Also save summary CSV
    summary_data = []
    for run in runs:
        row = {
            'rank': 0,
            'run_id': run['run_id'],
            'val_dir_acc': run['val_directional_accuracy'],
            'test_dir_acc': run.get('final_test_directional_accuracy', 0),
            'val_mse': run['val_mse'],
            **{f"config_{k}": v for k, v in run['config'].items()}
        }
        summary_data.append(row)
    
    # Add ranks
    for i, row in enumerate(summary_data, 1):
        row['rank'] = i
    
    import pandas as pd
    df = pd.DataFrame(summary_data)
    csv_path = output_dir / "sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV: {csv_path}")
    
    return output_dir


def print_sweep_statistics(runs: List[Dict]):
    """Print summary statistics of the sweep."""
    if not runs:
        print("No completed runs found.")
        return
    
    print(f"\n{'='*60}")
    print("Sweep Statistics")
    print(f"{'='*60}")
    print(f"Total completed runs: {len(runs)}")
    
    # Extract key hyperparameters
    val_accs = [r['val_directional_accuracy'] for r in runs]
    test_accs = [r.get('final_test_directional_accuracy', 0) for r in runs if r.get('final_test_directional_accuracy')]
    
    print(f"\nValidation Directional Accuracy:")
    print(f"  Mean: {sum(val_accs)/len(val_accs):.4f}")
    print(f"  Std:  {pd.Series(val_accs).std():.4f}")
    print(f"  Min:  {min(val_accs):.4f}")
    print(f"  Max:  {max(val_accs):.4f}")
    
    if test_accs:
        print(f"\nTest Directional Accuracy:")
        print(f"  Mean: {sum(test_accs)/len(test_accs):.4f}")
        print(f"  Std:  {pd.Series(test_accs).std():.4f}")
        print(f"  Min:  {min(test_accs):.4f}")
        print(f"  Max:  {max(test_accs):.4f}")
    
    # Best configuration details
    best = runs[0]
    print(f"\n{'='*60}")
    print(f"Best Configuration (Run: {best['run_id']})")
    print(f"{'='*60}")
    print(f"Val Directional Accuracy: {best['val_directional_accuracy']:.4f}")
    print(f"Test Directional Accuracy: {best.get('final_test_directional_accuracy', 0):.4f}")
    print(f"\nHyperparameters:")
    
    important_params = ['model', 'commodity', 'horizon', 'hidden_dim', 'num_layers', 
                       'dropout', 'lr', 'batch_size', 'seq_len']
    for param in important_params:
        if param in best['config']:
            print(f"  {param}: {best['config'][param]}")
    
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Extract best configs from wandb sweep')
    parser.add_argument('--sweep_id', type=str, required=True,
                       help='Wandb sweep ID (e.g., crop-futures/abc123)')
    parser.add_argument('--entity', type=str, default=None,
                       help='Wandb entity/username (if not in sweep_id)')
    parser.add_argument('--project', type=str, default='crop-futures-prediction',
                       help='Wandb project name')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top configurations to export')
    parser.add_argument('--model', type=str, default=None,
                       help='Filter by model type')
    parser.add_argument('--min_val_acc', type=float, default=0.0,
                       help='Minimum validation accuracy to include')
    parser.add_argument('--output_dir', type=str, default='training/best_configs',
                       help='Directory to save best configs')
    
    args = parser.parse_args()
    
    # Fetch runs
    runs = get_sweep_results(args.sweep_id, args.entity, args.project)
    
    if not runs:
        print("No runs found. Exiting.")
        return
    
    # Filter and sort
    runs = filter_and_sort_runs(runs, args.model, args.min_val_acc)
    
    if not runs:
        print("No runs matching criteria. Exiting.")
        return
    
    # Print statistics
    print_sweep_statistics(runs)
    
    # Export best configs
    export_best_configs(runs, args.output_dir, args.top_k)
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print(f"1. Review exported configs in: {args.output_dir}")
    print(f"2. Train best model with full epochs:")
    print(f"   python3 training/train.py --model attention_vision_dual_stream \\")
    print(f"     --epochs 200 --no_early_stopping \\")
    print(f"     --hidden_dim <value> --dropout <value> --lr <value>")
    print(f"3. Or use the exported JSON directly in your training pipeline")


if __name__ == "__main__":
    main()
