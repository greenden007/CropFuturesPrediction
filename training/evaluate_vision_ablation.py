#!/usr/bin/env python3
"""
Evaluate image encoder contribution for VisionDualStreamLSTM checkpoints.

Runs the same test split under:
1) normal images
2) zeroed images
3) shuffled images (batch permutation)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import load_unified_data, get_feature_groups, create_dataloaders
from models import create_model


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))


def evaluate_mode(model, loader, device, mode: str):
    model.eval()
    criterion = nn.MSELoss()
    preds, targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for price_x, fund_x, vision_x, y in loader:
            price_x = price_x.to(device)
            fund_x = fund_x.to(device)
            y = y.to(device)
            vision_x = vision_x.to(device)

            if mode == "zeros":
                vision_x = torch.zeros_like(vision_x)
            elif mode == "shuffled":
                perm = torch.randperm(vision_x.size(0), device=vision_x.device)
                vision_x = vision_x[perm]

            out = model(price_x, fund_x, vision_x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds.extend(out.cpu().numpy().reshape(-1))
            targets.extend(y.cpu().numpy().reshape(-1))

    preds = np.array(preds)
    targets = np.array(targets)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    return {
        "loss": float(total_loss / max(1, len(loader))),
        "mse": mse,
        "mae": mae,
        "directional_acc": directional_accuracy(targets, preds)
    }


def main():
    parser = argparse.ArgumentParser(description="Vision encoder ablation evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to *_best.pt checkpoint")
    parser.add_argument("--data", type=str, default="merged_data/daily_unified.csv")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    if config.get("model_type") != "dual_stream_lstm":
        raise ValueError("Checkpoint must be vision_dual_stream_lstm")

    df = load_unified_data(args.data)
    target_col = config["target_col"]
    groups = get_feature_groups(df)
    feature_cols = [c for c in sum(groups.values(), []) if c != target_col]

    train_loader, val_loader, test_loader = create_dataloaders(
        df, feature_cols, target_col,
        seq_len=config["seq_len"],
        pred_horizon=config["pred_horizon"],
        batch_size=config["batch_size"],
        train_split=config["train_split"],
        val_split=config["val_split"],
        model_type="dual_stream_lstm",
        outlook_index_path="processed_data/weather_outlooks/weather_outlook_index.csv",
        data_dir="data/noaa_cpc_discussions/cpc_outlook_data"
    )

    price_cols = [c for c in groups["price"] if c != target_col]
    price_dim = len(price_cols) + len(groups["volume"])
    fund_dim = len(groups["wasde"]) + len(groups["crop_progress"]) + len(groups["weather"]) + len(groups["image"]) + len(groups["time"])

    model = create_model(
        "dual_stream_lstm",
        price_input_dim=price_dim,
        fund_input_dim=fund_dim,
        vision_feature_dim=64,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        output_dim=1,
        dropout=config["dropout"],
        num_images=6
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    results = {
        "normal": evaluate_mode(model, test_loader, device, "normal"),
        "zeros": evaluate_mode(model, test_loader, device, "zeros"),
        "shuffled": evaluate_mode(model, test_loader, device, "shuffled")
    }
    results["delta_mse_zero_minus_normal"] = results["zeros"]["mse"] - results["normal"]["mse"]
    results["delta_mse_shuffle_minus_normal"] = results["shuffled"]["mse"] - results["normal"]["mse"]

    out_path = Path(args.output) if args.output else ckpt_path.with_name(ckpt_path.stem.replace("_best", "_vision_ablation") + ".json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
