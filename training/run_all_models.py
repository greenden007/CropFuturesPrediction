#!/usr/bin/env python3
"""
Run All Models Script with Weights & Biases Integration

Runs individual commodity models with different architectures and horizons.
Logs all results to Weights & Biases for easy comparison.

Usage:
    python run_all_models.py --wandb_project crop_futures
    python run_all_models.py --commodities corn wheat --horizons 1 5 --wandb_project crop_futures
"""

import os
import sys
import json
import argparse
import itertools
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# Default hyperparameters per model type
DEFAULT_MODEL_CONFIGS = {
    'gru': {
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.001,
        'hidden_dim': 128,
    },
    'resnet': {
        'epochs': 150,
        'batch_size': 64,
        'lr': 0.0005,
        'hidden_dim': 64,
    },
    'transformer': {
        'epochs': 200,
        'batch_size': 32,
        'lr': 0.0001,
        'hidden_dim': 256,
    },
    'tft': {
        'epochs': 200,
        'batch_size': 32,
        'lr': 0.0003,
        'hidden_dim': 128,
    },
    'patchtst': {
        'epochs': 200,
        'batch_size': 32,
        'lr': 0.0002,
        'hidden_dim': 128,
    },
    'nbeats': {
        'epochs': 200,
        'batch_size': 64,
        'lr': 0.0005,
        'hidden_dim': 256,
    },
    'dual_stream_lstm': {
        'epochs': 120,
        'batch_size': 32,
        'lr': 0.001,
        'hidden_dim': 128,
    }
}


def run_training(commodity, model, horizon, epochs, batch_size, lr, hidden_dim, 
                 output_dir, data_path, use_wandb, wandb_project, wandb_entity):
    """Run a single training job."""
    
    model_name = f"{model}_{commodity}_h{horizon}"
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"  Commodity: {commodity}")
    print(f"  Model: {model}")
    print(f"  Horizon: {horizon}")
    print(f"{'='*70}")
    
    # Build command
    cmd = [
        'python', 'training/train.py',
        '--model', model,
        '--commodity', commodity,
        '--horizon', str(horizon),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--hidden_dim', str(hidden_dim),
        '--output_dir', str(output_dir),
        '--data', data_path
    ]
    
    # Run training
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Print output for debugging if failed
    if result.returncode != 0:
        print(f"\n✗ FAILED - {commodity}/{model}/h{horizon}")
        if result.stdout:
            print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    
    # Parse results from output directory
    results_file = Path(output_dir) / f"{model_name}_results.json"
    metrics = {
        'commodity': commodity,
        'model': model,
        'horizon': horizon,
        'duration_seconds': duration,
        'success': result.returncode == 0
    }
    
    if results_file.exists():
        try:
            with open(results_file) as f:
                saved_results = json.load(f)
            if 'test_metrics' in saved_results:
                test_metrics = saved_results['test_metrics']
                metrics.update({
                    'test_mse': test_metrics.get('mse', float('inf')),
                    'test_mae': test_metrics.get('mae', float('inf')),
                    'test_directional_acc': test_metrics.get('directional_acc', 0.0),
                    'test_rmse': test_metrics.get('mse', 0) ** 0.5 if test_metrics.get('mse') else float('inf')
                })
            if 'config' in saved_results:
                config = saved_results['config']
                metrics['n_parameters'] = config.get('n_parameters', 0)
            if 'n_parameters' in saved_results:
                metrics['n_parameters'] = saved_results.get('n_parameters', metrics.get('n_parameters', 0))
        except json.JSONDecodeError as e:
            metrics['error'] = f'Corrupted results file: {e}'
    else:
        metrics['error'] = 'Results file not found'
        if result.stderr:
            metrics['stderr'] = result.stderr[-500:]  # Last 500 chars
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=model_name,
            config={
                'commodity': commodity,
                'model': model,
                'horizon': horizon,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'hidden_dim': hidden_dim
            },
            reinit=True
        )
        
        # Log metrics
        wandb.log({
            'test_mse': metrics.get('test_mse', float('inf')),
            'test_mae': metrics.get('test_mae', float('inf')),
            'test_rmse': metrics.get('test_rmse', float('inf')),
            'test_directional_acc': metrics.get('test_directional_acc', 0.0),
            'duration_seconds': duration,
            'success': metrics['success']
        })
        
        # Log model file as artifact
        model_file = Path(output_dir) / f"{model_name}_best.pt"
        if model_file.exists():
            artifact = wandb.Artifact(f"model-{model_name}", type="model")
            artifact.add_file(str(model_file))
            wandb.log_artifact(artifact)
        
        run.finish()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run all commodity models with wandb tracking')
    
    # Commodities to train
    parser.add_argument('--commodities', nargs='+', default=['corn', 'soybeans', 'wheat'],
                       choices=['corn', 'soybeans', 'wheat'],
                       help='Commodities to train on')
    
    # Models to train
    parser.add_argument('--models', nargs='+', 
                       default=['gru', 'resnet', 'transformer', 'patchtst', 'nbeats', 'tft', 'dual_stream_lstm'],
                       choices=['gru', 'resnet', 'transformer', 'patchtst', 'nbeats', 'tft', 'dual_stream_lstm'],
                       help='Model architectures to train')
    
    # Prediction horizons
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 5],
                       help='Prediction horizons in days')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    
    # Data and output (paths resolved from repository root)
    parser.add_argument('--data', type=str, default='merged_data/daily_unified.csv',
                       help='Path to unified data (from repository root)')
    parser.add_argument('--output_dir', type=str, default='results_all_models',
                       help='Output directory')
    
    # Wandb settings
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='crop_futures_prediction',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity/team name')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                       help='Wandb API key (or set WANDB_API_KEY env var/.env file)')
    
    # Per-model hyperparameter configs
    parser.add_argument('--use_model_configs', action='store_true',
                       help='Use default hyperparameters tailored for each model type (overrides global settings)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to JSON file with model-specific hyperparameters')
    
    # Control options
    parser.add_argument('--dry_run', action='store_true',
                       help='Print what would be run without executing')
    parser.add_argument('--continue_on_error', action='store_true',
                       help='Continue training if one model fails (default: stop on error)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run models in parallel (not recommended for local machines)')
    
    args = parser.parse_args()
    
    # Load .env file if available
    if DOTENV_AVAILABLE:
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded .env from {env_path}")
        else:
            # Try current directory
            load_dotenv()
    
    # Get wandb API key (priority: arg > env var > .env)
    wandb_api_key = args.wandb_api_key or os.environ.get('WANDB_API_KEY')
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key
    
    # Load model configs from file if provided
    model_configs = DEFAULT_MODEL_CONFIGS.copy()
    if args.config_file:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            model_configs.update(json.load(f))
        print(f"✓ Loaded model configs from {args.config_file}")
    
    # Setup wandb
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        # Login check
        try:
            wandb.login()
            print(f"✓ Logged in to wandb, project: {args.wandb_project}")
        except Exception as e:
            print(f"✗ Wandb login failed: {e}")
            use_wandb = False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    combinations = list(itertools.product(args.commodities, args.models, args.horizons))
    
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Commodities: {args.commodities}")
    print(f"Models: {args.models}")
    print(f"Horizons: {args.horizons}")
    print(f"Total runs: {len(combinations)}")
    
    if args.use_model_configs:
        print(f"\nPer-model hyperparameters:")
        for model in args.models:
            cfg = model_configs.get(model, {})
            print(f"  {model}: epochs={cfg.get('epochs')}, batch={cfg.get('batch_size')}, "
                  f"lr={cfg.get('lr')}, hidden={cfg.get('hidden_dim')}")
    else:
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Hidden dim: {args.hidden_dim}")
    
    print(f"Wandb enabled: {use_wandb}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    if args.dry_run:
        print("DRY RUN - Would execute:")
        for i, (commodity, model, horizon) in enumerate(combinations, 1):
            print(f"  {i}. {model} on {commodity} (horizon={horizon})")
        return
    
    # Run all combinations
    all_results = []
    failed_runs = []
    
    for i, (commodity, model, horizon) in enumerate(combinations, 1):
        print(f"\n\n>>> Run {i}/{len(combinations)}")
        
        # Get hyperparameters for this model
        if args.use_model_configs and model in model_configs:
            cfg = model_configs[model]
            epochs = cfg.get('epochs', args.epochs)
            batch_size = cfg.get('batch_size', args.batch_size)
            lr = cfg.get('lr', args.lr)
            hidden_dim = cfg.get('hidden_dim', args.hidden_dim)
        else:
            epochs = args.epochs
            batch_size = args.batch_size
            lr = args.lr
            hidden_dim = args.hidden_dim
        
        try:
            metrics = run_training(
                commodity=commodity,
                model=model,
                horizon=horizon,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                hidden_dim=hidden_dim,
                output_dir=args.output_dir,
                data_path=args.data,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity
            )
            
            all_results.append(metrics)
            
            # Print summary
            status = "✓ SUCCESS" if metrics.get('success') else "✗ FAILED"
            print(f"\n{status} - {metrics.get('commodity')}/{metrics.get('model')}/h{metrics.get('horizon')}")
            if 'test_mse' in metrics:
                print(f"  MSE: {metrics['test_mse']:.6f}, MAE: {metrics['test_mae']:.6f}, Dir Acc: {metrics['test_directional_acc']:.4f}")
            
        except Exception as e:
            print(f"\n✗ EXCEPTION: {e}")
            failed_runs.append({
                'commodity': commodity,
                'model': model,
                'horizon': horizon,
                'error': str(e)
            })
            
            if not args.continue_on_error:
                raise
    
    # Save summary
    summary = {
        'config': {
            'commodities': args.commodities,
            'models': args.models,
            'horizons': args.horizons,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'hidden_dim': args.hidden_dim,
            'use_model_configs': args.use_model_configs,
            'model_configs': model_configs if args.use_model_configs else None
        },
        'results': all_results,
        'failed_runs': failed_runs,
        'total_runs': len(combinations),
        'successful_runs': len([r for r in all_results if r.get('success')]),
        'failed_count': len(failed_runs)
    }
    
    summary_file = output_dir / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total runs: {len(combinations)}")
    print(f"Successful: {summary['successful_runs']}")
    print(f"Failed: {summary['failed_count']}")
    print(f"\nResults saved to: {summary_file}")
    
    if use_wandb:
        print(f"\nView results at: https://wandb.ai/{args.wandb_entity or 'user'}/{args.wandb_project}")
    
    # Print best models by commodity
    print(f"\n{'='*70}")
    print(f"BEST MODELS BY COMMODITY (lowest MSE)")
    print(f"{'='*70}")
    
    for commodity in args.commodities:
        commodity_results = [r for r in all_results 
                          if r.get('commodity') == commodity and r.get('success') and 'test_mse' in r]
        if commodity_results:
            best = min(commodity_results, key=lambda x: x.get('test_mse', float('inf')))
            print(f"{commodity:>10}: {best['model']:<20} (h={best['horizon']}) - MSE: {best['test_mse']:.6f}")


if __name__ == "__main__":
    main()
