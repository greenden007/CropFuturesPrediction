import numpy as np
import pandas as pd

class SignalGenerator:
    """Converts continuous expected returns to discrete trading signals."""
    def __init__(self, rolling_window=60, threshold_multiplier=0.75):
        self.rolling_window = rolling_window
        self.threshold_multiplier = threshold_multiplier
        self.history = []

    def process_prediction(self, predicted_return):
        """
        Takes a new prediction, updates the rolling history, 
        and returns Long (+1), Short (-1), or Hold (0).
        """
        self.history.append(predicted_return)
        
        if len(self.history) < self.rolling_window:
            return 0 # Not enough data for statistical thresholding
        
        recent_preds = self.history[-self.rolling_window:]
        std_dev = np.std(recent_preds)
        mean_pred = np.mean(recent_preds)
        
        upper_bound = mean_pred + (std_dev * self.threshold_multiplier)
        lower_bound = mean_pred - (std_dev * self.threshold_multiplier)
        
        if predicted_return > upper_bound:
            return 1  # Long
        elif predicted_return < lower_bound:
            return -1 # Short
        else:
            return 0  # Hold
