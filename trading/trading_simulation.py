"""
Trading Strategy Simulation

Evaluates model predictions in a realistic trading context:
- Direction-based positions (long/short)
- Transaction costs
- Risk management (stop-loss, position sizing)
- Performance metrics: Sharpe ratio, max drawdown, win rate

Usage:
    python trading_simulation.py --model dual_stream_lstm --commodity corn --horizon 1
    python trading_simulation.py --results_dir ../training/results_all_models
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from dataset import load_unified_data


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: datetime
    exit_date: datetime
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'end'


class TradingStrategy:
    """
    Simple directional trading strategy based on model predictions.
    """
    
    def __init__(
        self,
        threshold: float = 0.0,  # Minimum predicted return to trade
        stop_loss: float = 0.02,   # 2% stop loss
        take_profit: float = 0.05,  # 5% take profit
        transaction_cost: float = 0.001,  # 0.1% per trade
        max_position: float = 0.1,  # 10% of capital per trade
        fixed_position_size: float = 10000.0  # Fixed $10k per trade (overrides max_position if > 0)
    ):
        self.threshold = threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.fixed_position_size = fixed_position_size
        
        self.trades: List[Trade] = []
        self.position = 0  # -1, 0, or 1
        self.entry_price = 0.0
        self.entry_date = None
        self.position_value = 0.0  # Dollar amount per trade
        
    def generate_signals(
        self,
        predictions: np.ndarray,
        current_prices: np.ndarray,
        target_prices: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Generate trading signals from predictions.

        Signal logic:
        - predicted_return > threshold: go long (1)
        - predicted_return < -threshold: go short (-1)
        - otherwise: neutral (0)

        Args:
            predictions: Model's predicted prices at target dates
            current_prices: Prices at time of prediction (no look-ahead)
            target_prices: Actual prices at target dates (for reference only)
            dates: Dates for each prediction
        """
        # Calculate predicted returns: predicted % change from CURRENT price (not future actual)
        # current_prices[i] is price at time of prediction for target date i
        pred_returns = (predictions - current_prices) / current_prices

        signals = np.where(
            pred_returns > self.threshold, 1,
            np.where(pred_returns < -self.threshold, -1, 0)
        )
        
        return pd.DataFrame({
            'date': dates,
            'prediction': predictions,
            'current_price': current_prices,
            'target_price': target_prices,
            'predicted_return': pred_returns,
            'signal': signals,
            'actual_return': np.concatenate([[0], np.diff(target_prices) / target_prices[:-1]])
        })
    
    def simulate(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Run trading simulation with risk management.
        """
        capital = initial_capital
        equity_curve = [capital]
        self.trades = []
        self.position = 0
        self.position_value = 0.0
        
        for i in range(1, len(signals_df)):
            date = signals_df['date'].iloc[i]
            price = signals_df['target_price'].iloc[i]
            signal = signals_df['signal'].iloc[i]
            
            # Check exit conditions if in position
            if self.position != 0:
                raw_pnl_pct = (price - self.entry_price) / self.entry_price * self.position
                
                # Handle gaps: if price gaps through stop/take levels, use the stop/take level
                if self.position == 1:  # Long
                    effective_pnl_pct = max(-self.stop_loss, min(self.take_profit, raw_pnl_pct))
                else:  # Short
                    effective_pnl_pct = max(-self.stop_loss, min(self.take_profit, raw_pnl_pct))
                
                exit_reason = None
                if raw_pnl_pct <= -self.stop_loss:
                    exit_reason = 'stop_loss'
                elif raw_pnl_pct >= self.take_profit:
                    exit_reason = 'take_profit'
                elif signal != self.position:  # Signal flip
                    exit_reason = 'signal'
                elif i == len(signals_df) - 1:  # End of data
                    exit_reason = 'end'
                
                # Use effective (capped) pnl_pct for calculation
                pnl_pct = effective_pnl_pct if exit_reason in ['stop_loss', 'take_profit'] else raw_pnl_pct
                
                if exit_reason:
                    # Close position - PnL based on fixed position value
                    pnl = pnl_pct * self.position_value
                    capital += pnl - (self.position_value * self.transaction_cost * 2)  # Entry + exit costs
                    
                    self.trades.append(Trade(
                        entry_date=self.entry_date,
                        exit_date=date,
                        direction=self.position,
                        entry_price=self.entry_price,
                        exit_price=price,
                        pnl=pnl,
                        pnl_pct=pnl_pct * 100,
                        exit_reason=exit_reason
                    ))
                    
                    self.position = 0
                    self.position_value = 0.0
            
            # Enter new position if neutral and signal exists
            if self.position == 0 and signal != 0:
                self.position = signal
                self.entry_price = price
                self.entry_date = date
                # Use fixed position size if specified (strictly enforced), otherwise % of capital
                if self.fixed_position_size > 0:
                    self.position_value = min(self.fixed_position_size, capital * 0.5)  # Hard cap at 50% of capital max
                else:
                    self.position_value = min(capital * self.max_position, capital * 0.5)  # Cap at 50% of capital
                capital -= self.position_value * self.transaction_cost  # Entry cost only
            
            equity_curve.append(capital)
        
        return self._calculate_metrics(equity_curve, initial_capital)
    
    def _calculate_metrics(
        self,
        equity_curve: List[float],
        initial_capital: float
    ) -> Dict:
        """Calculate trading performance metrics."""
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Basic metrics
        total_return = (equity[-1] - initial_capital) / initial_capital
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Trade statistics
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            win_rate = np.mean([1 if t.pnl > 0 else 0 for t in self.trades])
            avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if any(t.pnl > 0 for t in self.trades) else 0
            avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if any(t.pnl < 0 for t in self.trades) else 0
            profit_factor = abs(sum(t.pnl for t in self.trades if t.pnl > 0)) / abs(sum(t.pnl for t in self.trades if t.pnl < 0)) if sum(t.pnl < 0 for t in self.trades) > 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': equity[-1],
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown * 100,
            'n_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity.tolist(),
            'trades': [
                {
                    'entry_date': str(t.entry_date),
                    'exit_date': str(t.exit_date),
                    'direction': 'long' if t.direction == 1 else 'short',
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'exit_reason': t.exit_reason
                }
                for t in self.trades
            ]
        }


def load_model_predictions(results_dir: Path, model_name: str, commodity: str, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions and actuals from model results."""
    results_file = results_dir / f"{model_name}_{commodity}_h{horizon}_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Load test predictions
    test_metrics = data.get('test_metrics', {})
    predictions = np.array(test_metrics.get('predictions', []))
    actuals = np.array(test_metrics.get('targets', []))
    
    if len(predictions) == 0 or len(actuals) == 0:
        raise ValueError(f"No predictions found in {results_file}. Need to re-run training with updated train.py")
    
    return predictions, actuals


def run_trading_simulation(
    model_name: str,
    commodity: str,
    horizon: int,
    results_dir: Path,
    data_path: Path,
    strategy_params: Dict
) -> Dict:
    """
    Run complete trading simulation for a model.
    """
    print(f"\n{'='*60}")
    print(f"TRADING SIMULATION: {model_name} on {commodity} (H={horizon})")
    print(f"{'='*60}")
    
    # Load data
    df = load_unified_data(data_path)
    
    # Get test period dates
    target_col = f'{commodity[:3]}_close'
    df = df.sort_values('date')
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[val_end:] = True
    
    test_df = df[test_mask].copy()
    
    # Load from saved model results
    try:
        predictions, actuals = load_model_predictions(
            results_dir, model_name, commodity, horizon
        )

        # Align with dates - use the end of the series (test period)
        if len(predictions) != len(test_df):
            print(f"Note: Length mismatch. Predictions: {len(predictions)}, Test data: {len(test_df)}")
            # Assume predictions match the last N test samples
            if len(predictions) < len(test_df):
                test_df = test_df.iloc[-len(predictions):]
            else:
                predictions = predictions[-len(test_df):]
                actuals = actuals[-len(test_df):]

    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading predictions: {e}")
        return {
            'model': model_name,
            'commodity': commodity,
            'horizon': horizon,
            'error': str(e)
        }

    # Get current prices at prediction time (H days before target dates)
    # This is crucial to avoid look-ahead bias in signal generation
    target_col = f'{commodity[:3]}_close'
    test_prices = test_df[target_col].values

    # For prediction at index i (targeting date i), the prediction was made H days earlier
    # So current_price[i] = price at date i - H
    # For the first H predictions, we need prices from before the test period
    current_prices = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if i >= horizon:
            # Use price from H days earlier within test period
            current_prices[i] = test_prices[i - horizon]
        else:
            # For early predictions, use the earliest available test price
            # (approximation - ideally would use validation period prices)
            current_prices[i] = test_prices[0]

    # Run strategy
    strategy = TradingStrategy(**strategy_params)
    signals_df = strategy.generate_signals(
        predictions, current_prices, actuals, pd.to_datetime(test_df['date'])
    )
    
    results = strategy.simulate(signals_df)

    # Calculate buy-and-hold benchmark for same period
    test_prices_benchmark = test_df[target_col].values
    bh_returns = np.diff(test_prices_benchmark) / test_prices_benchmark[:-1]
    bh_sharpe = np.mean(bh_returns) / np.std(bh_returns) * np.sqrt(252) if len(bh_returns) > 1 and np.std(bh_returns) > 0 else 0.0
    bh_total_return = (test_prices_benchmark[-1] - test_prices_benchmark[0]) / test_prices_benchmark[0] * 100

    # Add benchmark to results
    results['buy_hold_sharpe'] = bh_sharpe
    results['buy_hold_return_pct'] = bh_total_return
    results['sharpe_improvement'] = results['sharpe_ratio'] - bh_sharpe

    # Print summary
    if 'error' in results:
        print(f"  Error: {results['error']}")
    else:
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {results['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"  Number of Trades: {results['n_trades']}")
        print(f"  Win Rate: {results['win_rate']:.2%}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Final Capital: ${results['final_capital']:,.2f}")
        print(f"\nBenchmark (Buy & Hold):")
        print(f"  Total Return: {bh_total_return:.2f}%")
        print(f"  Sharpe Ratio: {bh_sharpe:.3f}")
        print(f"  Improvement: {results['sharpe_improvement']:+.3f}")

    return results


def compare_strategies(
    models: List[str],
    commodity: str,
    horizon: int,
    results_dir: Path,
    data_path: Path
) -> pd.DataFrame:
    """
    Compare trading performance across multiple models.
    """
    strategy_params = {
        'threshold': 0.0,
        'stop_loss': 0.02,
        'take_profit': 0.05,
        'transaction_cost': 0.001
    }
    
    all_results = []
    
    for model in models:
        try:
            results = run_trading_simulation(
                model, commodity, horizon, results_dir, data_path, strategy_params
            )
            results['model'] = model
            all_results.append(results)
        except Exception as e:
            print(f"Error running {model}: {e}")
    
    # Create comparison table
    comparison = pd.DataFrame([
        {
            'Model': r['model'],
            'Return %': f"{r['total_return_pct']:.2f}",
            'Sharpe': f"{r['sharpe_ratio']:.3f}",
            'Max DD %': f"{r['max_drawdown_pct']:.2f}",
            'Trades': r['n_trades'],
            'Win Rate': f"{r['win_rate']:.2%}",
            'Profit Factor': f"{r['profit_factor']:.2f}"
        }
        for r in all_results
    ])
    
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(comparison.to_string(index=False))
    
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading strategy simulation')
    
    parser.add_argument('--model', type=str, default='dual_stream_lstm',
                       help='Model to evaluate')
    parser.add_argument('--commodity', type=str, default='corn',
                       choices=['corn', 'soybeans', 'wheat'])
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--results_dir', type=str,
                       default='../training/results_all_models',
                       help='Directory with model prediction results (relative to trading/ dir)')
    parser.add_argument('--data', type=str, default='../merged_data/daily_unified.csv')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Prediction threshold for trading')
    parser.add_argument('--stop_loss', type=float, default=0.02,
                       help='Stop loss percentage')
    parser.add_argument('--take_profit', type=float, default=0.05,
                       help='Take profit percentage')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                       help='Transaction cost per trade (0.001 = 0.1%)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    parser.add_argument('--models', nargs='+',
                       default=['dual_stream_lstm', 'gru', 'transformer', 'resnet'],
                       help='Models to compare')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    data_path = Path(args.data)
    
    if args.compare:
        compare_strategies(
            args.models, args.commodity, args.horizon, results_dir, data_path
        )
    else:
        strategy_params = {
            'threshold': args.threshold,
            'stop_loss': args.stop_loss,
            'take_profit': args.take_profit,
            'transaction_cost': args.transaction_cost
        }
        
        results = run_trading_simulation(
            args.model, args.commodity, args.horizon,
            results_dir, data_path, strategy_params
        )
        
        # Save results
        output_file = Path('trading_results') / f'{args.model}_{args.commodity}_h{args.horizon}_trading.json'
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
