import pandas as pd
import numpy as np
from agent.metrics import sharpe_ratio

class BacktestSimulator:
    """A minimal simulator to track portfolio value over time."""
    def __init__(self, initial_capital=100000.0, commission=0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.position = 0 # +1 Long, -1 Short, 0 Flat
        self.portfolio_history = []
        
    def step(self, signal, actual_return):
        """
        Advances the simulator by one step.
        signal: +1, -1, or 0
        actual_return: The actual market return for that period.
        """
        # Calculate PnL from previous position
        if self.position != 0:
            # We assume position sizing is 100% of capital for simplicity
            pnl = self.capital * self.position * actual_return
            self.capital += pnl
            
        # Check if we are changing position and need to pay commission
        if signal != self.position:
            # Deduct commission based on capital sizing
            self.capital -= self.capital * self.commission
            self.position = signal
            
        self.portfolio_history.append(self.capital)
        
    def summary(self):
        history = pd.Series(self.portfolio_history)
        returns = history.pct_change().dropna()
        
        if len(returns) == 0:
            return {"Total Return": 0, "Sharpe Ratio": 0, "Max Drawdown": 0}
            
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        sharpe = sharpe_ratio(returns)  # Shared metric — see agent/metrics.py
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown
        }
