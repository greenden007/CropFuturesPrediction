"""
agent/metrics.py
----------------
Shared evaluation metrics for the CropFuturesPrediction pipeline.

Import this module anywhere you need a standardised performance metric so that
all team members are computing the same numbers:

    from agent.metrics import sharpe_ratio
"""

import numpy as np
import pandas as pd
from typing import Union


def sharpe_ratio(
    returns: Union[pd.Series, np.ndarray, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the annualised Sharpe ratio for a sequence of periodic returns.

    This is the **canonical shared metric** for this project.  Every model /
    strategy variant should report this number so results are directly
    comparable across team members.

    Parameters
    ----------
    returns : array-like
        Sequence of per-period (e.g. daily) **arithmetic** returns.
        Pass in raw returns, *not* log-returns or cumulative values.
    risk_free_rate : float, optional
        Annualised risk-free rate (default 0.0).  Converted internally to a
        per-period rate before subtraction.
    periods_per_year : int, optional
        Number of trading periods in a year (default 252 for daily data;
        use 52 for weekly, 12 for monthly, etc.).

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns 0.0 when there are fewer than two
        observations or when volatility is exactly zero.

    Notes
    -----
    Formula::

        Sharpe = (mean(r - rf_period) / std(r, ddof=1)) * sqrt(periods_per_year)

    where ``rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1``.

    Examples
    --------
    >>> from agent.metrics import sharpe_ratio
    >>> import numpy as np
    >>> daily_returns = np.random.normal(0.0005, 0.01, 252)
    >>> sharpe_ratio(daily_returns)
    """
    returns = np.asarray(returns, dtype=float)

    # Need at least two data points to compute a standard deviation
    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to per-period rate
    rf_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_period

    std = excess.std(ddof=1)
    if std < 1e-12:
        return 0.0

    return float((excess.mean() / std) * np.sqrt(periods_per_year))
