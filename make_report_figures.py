from __future__ import annotations

import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PRICE_COLS = ["corn", "soybeans", "wheat"]


def _safe_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def featurize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index, list[str]]:
    feat_blocks = []
    feat_names = []

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


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    try:
        return pd.read_parquet(path).sort_index()
    except Exception:
        fallback = path.with_suffix(".csv")
        if fallback.exists():
            return pd.read_csv(fallback, index_col=0, parse_dates=True).sort_index()
        raise


def compute_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()
    signal = pred_df["signal"].to_numpy()
    strategy_return = pred_df["strategy_return"].to_numpy()

    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    direction_acc = float(np.mean((y_pred > 0) == (y_true > 0)))
    baseline_zero_rmse = math.sqrt(np.mean(np.square(y_true)))

    equity = np.cumprod(1.0 + strategy_return)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    ret_std = np.std(strategy_return, ddof=1) if len(strategy_return) > 1 else 0.0
    sharpe = float(np.mean(strategy_return) / ret_std * math.sqrt(13)) if ret_std > 0 else 0.0

    active = signal != 0
    win_rate = float(np.mean(strategy_return[active] > 0)) if np.any(active) else 0.0
    num_active = int(np.sum(active))

    strategy_cumulative_return = float(np.prod(1.0 + strategy_return) - 1.0)
    buy_hold_return = float(np.prod(1.0 + y_true) - 1.0)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Direction Acc": direction_acc,
        "Zero Baseline RMSE": baseline_zero_rmse,
        "Strategy Cum Return": strategy_cumulative_return,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Active Periods": num_active,
        "Buy & Hold Return": buy_hold_return,
    }


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values("week_start").reset_index(drop=True)


def make_equity_curves(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["strategy_equity"] = (1.0 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = (1.0 + out["y_true"]).cumprod()
    return out


def save_metrics_table_png(metrics_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.axis("off")

    display_df = metrics_df.copy()
    for col in display_df.columns[1:]:
        if col == "Active Periods":
            display_df[col] = display_df[col].astype(int).astype(str)
        else:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_equity_comparison(ridge_df: pd.DataFrame, rf_df: pd.DataFrame, out_path: Path):
    ridge_eq = make_equity_curves(ridge_df)
    rf_eq = make_equity_curves(rf_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ridge_eq["week_start"], ridge_eq["strategy_equity"], label="Ridge Strategy")
    ax.plot(rf_eq["week_start"], rf_eq["strategy_equity"], label="RF Strategy")
    ax.plot(rf_eq["week_start"], rf_eq["buy_hold_equity"], label="Buy & Hold")
    ax.set_title("Test-Period Equity Curves")
    ax.set_xlabel("Week")
    ax.set_ylabel("Equity Growth")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_pred_time(pred_df: pd.DataFrame, model_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(pred_df["week_start"], pred_df["y_true"], label="Actual")
    ax.plot(pred_df["week_start"], pred_df["y_pred"], label="Predicted")
    ax.set_title(f"{model_name}: Actual vs Predicted 4-Week Returns")
    ax.set_xlabel("Week")
    ax.set_ylabel("Return")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(pred_df: pd.DataFrame, model_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pred_df["y_true"], pred_df["y_pred"], alpha=0.7)
    lo = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
    hi = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rf_feature_importance(dataset_path: Path, model_path: Path, out_path: Path, test_fraction: float = 0.2):
    df = load_dataset(dataset_path)
    X, y, idx, feat_names = featurize(df)

    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]

    top_names = [feat_names[i] for i in order][::-1]
    top_vals = [importances[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_names, top_vals)
    ax.set_title("Random Forest Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    base = Path(".")
    models_dir = base / "models"
    figs_dir = base / "report_figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = base / "data" / "dataset.parquet"
    if not dataset_path.exists():
        dataset_path = base / "data" / "dataset.csv"

    ridge_preds_path = models_dir / "ridge_test_predictions.csv"
    rf_preds_path = models_dir / "rf_test_predictions.csv"
    rf_model_path = models_dir / "rf.pkl"

    ridge_df = load_preds(ridge_preds_path)
    rf_df = load_preds(rf_preds_path)

    ridge_metrics = compute_metrics(ridge_df)
    rf_metrics = compute_metrics(rf_df)

    metrics_df = pd.DataFrame([
        {"Model": "Ridge", **ridge_metrics},
        {"Model": "Random Forest", **rf_metrics},
    ])

    metrics_df.to_csv(figs_dir / "metrics_summary.csv", index=False)

    save_metrics_table_png(metrics_df, figs_dir / "metrics_summary_table.png")
    plot_equity_comparison(ridge_df, rf_df, figs_dir / "equity_curves.png")

    plot_actual_vs_pred_time(ridge_df, "Ridge", figs_dir / "ridge_actual_vs_pred_time.png")
    plot_actual_vs_pred_time(rf_df, "Random Forest", figs_dir / "rf_actual_vs_pred_time.png")

    plot_scatter(ridge_df, "Ridge", figs_dir / "ridge_scatter.png")
    plot_scatter(rf_df, "Random Forest", figs_dir / "rf_scatter.png")

    if rf_model_path.exists() and dataset_path.exists():
        plot_rf_feature_importance(dataset_path, rf_model_path, figs_dir / "rf_feature_importance.png")

    print("Saved figures to:", figs_dir.resolve())
    print("Saved files:")
    for p in sorted(figs_dir.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()