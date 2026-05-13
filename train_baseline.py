from __future__ import annotations

"""
Train simple baseline models on the preprocessed weekly dataset and evaluate
both forecasting metrics and trading-style metrics.

Recommended first run:
    python train_baseline.py --dataset ./data/dataset.parquet --mode rf

Modes:
  - ridge: fast linear baseline
  - rf:    RandomForestRegressor baseline
  - mlp:   simple PyTorch MLP (if torch is installed)

Outputs:
  - regression metrics: RMSE, MAE, R2
  - directional metric: direction_acc
  - trading metrics: cumulative return, Sharpe ratio, max drawdown, win rate
"""

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PRICE_COLS = ["corn", "soybeans", "wheat"]


def _safe_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def featurize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index, list[str]]:
    if "return_4w" not in df.columns:
        raise ValueError("`return_4w` column not found in dataset")
    if "corn" not in df.columns:
        raise ValueError("`corn` column not found in dataset")

    feat_blocks: list[np.ndarray] = []
    feat_names: list[str] = []

    for col in PRICE_COLS:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")

        for lag in [1, 2, 4, 8]:
            vals = s.pct_change(periods=lag)
            feat_blocks.append(vals.to_numpy())
            feat_names.append(f"{col}_ret_lag_{lag}")

        ma4 = s.rolling(4).mean()
        ma12 = s.rolling(12).mean()
        momentum = (s / ma4) - 1.0
        trend = (ma4 / ma12) - 1.0
        vol4 = s.pct_change().rolling(4).std()

        feat_blocks.extend([momentum.to_numpy(), trend.to_numpy(), vol4.to_numpy()])
        feat_names.extend([f"{col}_mom_4", f"{col}_trend_4_12", f"{col}_vol_4"])

    crop_text = _safe_text_series(df, "text_crop_progress")
    cpc_text = _safe_text_series(df, "text_cpc")
    combined_text = crop_text + " " + cpc_text

    text_feats = {
        "crop_text_len": crop_text.str.len().to_numpy(),
        "cpc_text_len": cpc_text.str.len().to_numpy(),
        "total_text_len": combined_text.str.len().to_numpy(),
        "crop_word_count": crop_text.str.split().str.len().to_numpy(),
        "cpc_word_count": cpc_text.str.split().str.len().to_numpy(),
        "total_word_count": combined_text.str.split().str.len().to_numpy(),
    }

    for name, values in text_feats.items():
        feat_blocks.append(values)
        feat_names.append(name)

    X = np.column_stack(feat_blocks).astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = pd.to_numeric(df["return_4w"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y)
    X = X[keep]
    y = y[keep]
    idx = df.index[keep]

    return X, y, idx, feat_names


def train_ridge(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)
    return model


def train_rf(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_predictions(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        rmse = mean_squared_error(y_true, preds, squared=False)
    except TypeError:
        rmse = math.sqrt(mean_squared_error(y_true, preds))

    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    direction_acc = float(np.mean((preds > 0) == (y_true > 0)))
    baseline_zero_rmse = math.sqrt(np.mean(np.square(y_true)))

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "direction_acc": direction_acc,
        "baseline_zero_rmse": float(baseline_zero_rmse),
    }


def max_drawdown_from_returns(strategy_returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())


def backtest_from_predictions(
    y_true: np.ndarray,
    preds: np.ndarray,
    annualization: int = 13,
    threshold: float = 0.0,
    transaction_cost: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Very simple test-period strategy:
      signal = +1 if pred > threshold
      signal = -1 if pred < -threshold
      signal =  0 otherwise

    strategy_return_t = signal_t * realized_forward_return_t - cost_if_signal_changes

    Note:
    y_true here is already the realized 4-week forward return target.
    annualization=13 because there are roughly 13 four-week periods in a year.
    """
    signal = np.where(preds > threshold, 1, np.where(preds < -threshold, -1, 0)).astype(float)

    position_change = np.abs(np.diff(np.r_[0.0, signal]))
    costs = position_change * transaction_cost

    strategy_returns = signal * y_true - costs

    cumulative_return = float(np.prod(1.0 + strategy_returns) - 1.0)

    ret_std = np.std(strategy_returns, ddof=1) if len(strategy_returns) > 1 else 0.0
    if ret_std > 0:
        sharpe = float(np.mean(strategy_returns) / ret_std * math.sqrt(annualization))
    else:
        sharpe = 0.0

    max_dd = max_drawdown_from_returns(strategy_returns)

    active = signal != 0
    if np.any(active):
        win_rate = float(np.mean(strategy_returns[active] > 0))
        num_trades = int(np.sum(active))
    else:
        win_rate = 0.0
        num_trades = 0

    buy_hold_return = float(np.prod(1.0 + y_true) - 1.0)

    metrics = {
        "strategy_cumulative_return": cumulative_return,
        "strategy_sharpe": sharpe,
        "strategy_max_drawdown": max_dd,
        "strategy_win_rate": win_rate,
        "num_active_periods": num_trades,
        "buy_hold_return": buy_hold_return,
        "avg_signal": float(np.mean(signal)),
        "long_fraction": float(np.mean(signal == 1)),
        "short_fraction": float(np.mean(signal == -1)),
        "flat_fraction": float(np.mean(signal == 0)),
    }

    details = pd.DataFrame({
        "y_true": y_true,
        "y_pred": preds,
        "signal": signal,
        "strategy_return": strategy_returns,
        "cost": costs,
    })

    return metrics, details


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    preds = model.predict(X)
    metrics = evaluate_predictions(y, preds)
    return metrics, preds


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.suffix.lower() == ".csv":
        return pd.read_csv(dataset_path, index_col=0, parse_dates=True).sort_index()

    try:
        return pd.read_parquet(dataset_path).sort_index()
    except ImportError:
        fallback_csv = dataset_path.with_suffix(".csv")
        if fallback_csv.exists():
            print(f"Parquet engine not available; loaded CSV fallback: {fallback_csv}")
            return pd.read_csv(fallback_csv, index_col=0, parse_dates=True).sort_index()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/dataset.parquet")
    parser.add_argument("--mode", choices=["ridge", "rf", "mlp"], default="rf")
    parser.add_argument("--out", default="./models/model.pkl")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--save-preds", default="./models/test_predictions.csv")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}. Run data_prep.py first.")
        sys.exit(1)

    df = load_dataset(dataset_path)
    X, y, idx, feat_names = featurize(df)

    n = len(X)
    if n < 20:
        print(f"Dataset only has {n} rows after preprocessing. Get more data before training.")
        sys.exit(1)

    split = max(1, int(n * (1 - args.test_fraction)))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    idx_test = idx[split:]

    print(f"Rows: {n} | Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feat_names)}")

    if args.mode == "ridge":
        model = train_ridge(X_train, y_train)
        pred_metrics, preds = evaluate_model(model, X_test, y_test)

        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, outp)
        print(f"Saved model to {outp}")

    elif args.mode == "rf":
        model = train_rf(X_train, y_train)
        pred_metrics, preds = evaluate_model(model, X_test, y_test)

        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, outp)
        print(f"Saved model to {outp}")

        if hasattr(model, "feature_importances_"):
            order = np.argsort(model.feature_importances_)[::-1][:10]
            print("\nTop features:")
            for i in order:
                print(f"  {feat_names[i]}: {model.feature_importances_[i]:.4f}")

    else:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception:
            print("PyTorch not available; install torch or use --mode rf / --mode ridge")
            raise

        class MLP(nn.Module):
            def __init__(self, in_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        ds_train = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        dl = DataLoader(ds_train, batch_size=32, shuffle=True)

        for epoch in range(20):
            model.train()
            running = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
            print(f"Epoch {epoch + 1:02d} loss: {running / len(ds_train):.6f}")

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = model(X_test_t).cpu().numpy()

        pred_metrics = evaluate_predictions(y_test, preds)

    strategy_metrics, details = backtest_from_predictions(
        y_true=y_test,
        preds=preds,
        annualization=13,
        threshold=args.signal_threshold,
        transaction_cost=args.transaction_cost,
    )

    all_metrics = {**pred_metrics, **strategy_metrics}

    print("\nPrediction metrics:")
    for k, v in pred_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\nTrading metrics:")
    for k, v in strategy_metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    preview = pd.DataFrame({
        "week_start": idx_test.astype(str),
        "y_true": y_test,
        "y_pred": preds,
        "signal": details["signal"].values,
        "strategy_return": details["strategy_return"].values,
    })

    print("\nPrediction preview:")
    print(preview.head(10).to_string(index=False))

    save_preds = Path(args.save_preds)
    save_preds.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(save_preds, index=False)
    print(f"\nSaved test predictions to {save_preds}")


if __name__ == "__main__":
    main()