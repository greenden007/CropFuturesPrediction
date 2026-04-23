"""
Generate LaTeX tables for final results.
Usage: python generate_tables.py
"""

import json
from pathlib import Path

def generate_corn_tables():
    results_dir = Path('../training/results_all_models')
    trading_dir = Path('../trading/trading_results')
    commodity = 'corn'
    
    models = ['dual_stream_lstm', 'single_stream_lstm', 'transformer', 'gru', 'resnet']
    model_names = {
        'dual_stream_lstm': 'Dual-Stream LSTM',
        'single_stream_lstm': 'Single-Stream LSTM',
        'transformer': 'Transformer',
        'gru': 'GRU',
        'resnet': 'ResNet'
    }
    
    # Table 1: Test Performance Metrics
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{Test Performance Metrics for Corn Futures Prediction}')
    print(r'\label{tab:corn_performance}')
    print(r'\begin{tabular}{lccccc}')
    print(r'\toprule')
    print(r'Model & Horizon & MSE & MAE & Dir. Acc. & Params \\')
    print(r'\midrule')
    
    for model in models:
        for h in [1, 5]:
            results_file = results_dir / f'{model}_{commodity}_h{h}_results.json'
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                
                metrics = data['test_metrics']
                mse = metrics['mse']
                mae = metrics['mae']
                dir_acc = metrics['directional_acc']
                n_params = data.get('n_parameters', 0)
                
                model_name = model_names.get(model, model)
                print(f'{model_name} & H={h} & {mse:.6f} & {mae:.4f} & {dir_acc*100:.1f}\\% & {n_params:,} \\\\\\n', end='')
    
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')
    print()
    
    # Table 2: Trading Performance (H=1)
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{Trading Performance for Corn Futures (H=1, Fixed $10k Position)}')
    print(r'\label{tab:corn_trading}')
    print(r'\begin{tabular}{lccccc}')
    print(r'\toprule')
    print(r'Model & Return \% & Sharpe & Max DD \% & Win Rate & Profit Factor \\')
    print(r'\midrule')
    
    for model in models:
        trading_file = trading_dir / f'{model}_{commodity}_h1_trading.json'
        if trading_file.exists():
            with open(trading_file, encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip if results contain error
            if 'total_return_pct' not in data:
                continue
            
            ret = data['total_return_pct']
            sharpe = data['sharpe_ratio']
            max_dd = data['max_drawdown_pct']
            win_rate = data['win_rate']
            pf = data['profit_factor']
            
            model_name = model_names.get(model, model)
            print(f'{model_name} & {ret:.2f} & {sharpe:.3f} & {max_dd:.2f} & {win_rate*100:.1f}\\% & {pf:.2f} \\\\\\n', end='')
    
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')

if __name__ == '__main__':
    generate_corn_tables()
