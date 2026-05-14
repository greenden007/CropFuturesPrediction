#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_PATH="${DATA_PATH:-merged_data/daily_unified.csv}"
RESULTS_DIR="${RESULTS_DIR:-training/results_all_models}"
BASELINES_DIR="${BASELINES_DIR:-training/results_baselines_norm}"
TRADING_DIR="${TRADING_DIR:-trading/trading_results}"
TABLES_DIR="${TABLES_DIR:-tables}"

COMMODITIES="${COMMODITIES:-corn soybeans wheat}"
MODELS="${MODELS:-gru resnet transformer patchtst nbeats tft dual_stream_lstm}"
HORIZONS="${HORIZONS:-1 5 10}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
USE_MODEL_CONFIGS="${USE_MODEL_CONFIGS:-1}"

echo "============================================================"
echo "Crop Futures Full Pipeline"
echo "============================================================"
echo "Data:        $DATA_PATH"
echo "Results:     $RESULTS_DIR"
echo "Baselines:   $BASELINES_DIR"
echo "Trading:     $TRADING_DIR"
echo "Commodities: $COMMODITIES"
echo "Models:      $MODELS"
echo "Horizons:    $HORIZONS"
echo "============================================================"

mkdir -p "$RESULTS_DIR" "$BASELINES_DIR" "$TRADING_DIR" "$TABLES_DIR"

RUN_ALL_ARGS=(
  --commodities ${COMMODITIES}
  --models ${MODELS}
  --horizons ${HORIZONS}
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --hidden_dim "$HIDDEN_DIM"
  --data "$DATA_PATH"
  --output_dir "$RESULTS_DIR"
  --continue_on_error
)

if [[ "$USE_MODEL_CONFIGS" == "1" ]]; then
  RUN_ALL_ARGS+=(--use_model_configs)
fi

echo
echo "[1/5] Training all model combinations..."
python3 training/run_all_models.py "${RUN_ALL_ARGS[@]}"

echo
echo "[2/5] Running normalized classical baselines..."
for commodity in $COMMODITIES; do
  for horizon in $HORIZONS; do
    python3 training/classical_baselines.py \
      --commodity "$commodity" \
      --horizon "$horizon" \
      --data "$DATA_PATH" \
      --output_dir "$BASELINES_DIR" \
      --normalize_target
  done
done

echo
echo "[3/5] Ensuring predictions are present in model result JSON files..."
python3 training/extract_predictions.py \
  --all \
  --results_dir "$RESULTS_DIR" \
  --data "$DATA_PATH"

echo
echo "[4/5] Running trading simulation across all trained models..."
python3 trading/trading_simulation.py \
  --run_all \
  --results_dir "$RESULTS_DIR" \
  --data "$DATA_PATH" \
  --output_dir "$TRADING_DIR"

echo
echo "[5/5] Generating report tables and graphics..."
python3 tables/generate_trading_graphics.py \
  --csv "$TRADING_DIR/trading_comparison_summary.csv" \
  --output_dir "$TABLES_DIR"

echo
echo "Pipeline complete."
echo "Artifacts:"
echo "  - Training summary: $RESULTS_DIR/training_summary.json"
echo "  - Baselines:        $BASELINES_DIR"
echo "  - Trading summary:  $TRADING_DIR/trading_comparison_summary.csv"
echo "  - Tables/figures:   $TABLES_DIR"
