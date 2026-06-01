#!/bin/bash
#SBATCH --job-name=wandb_crop_futures_sweep
#SBATCH --output=logs/wandb_sweep_%A_%a.out
#SBATCH --error=logs/wandb_sweep_%A_%a.err
#SBATCH --array=1-20%4  # Run 20 jobs, 4 at a time
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu  # Adjust to your cluster's GPU partition

# Weights & Biases Sweep Runner for Crop Futures Prediction
# 
# This script runs parallel agents for a wandb hyperparameter sweep
# 
# Usage:
#   1. Initialize sweep: wandb sweep training/wandb_sweep.yaml
#   2. Set SWEEP_ID below to the ID from step 1
#   3. Submit: sbatch scripts/run_wandb_sweep.slurm
#
# To run a single experiment instead of a sweep:
#   sbatch --export=SWEEP_ID=none,MODEL=attention_vision_dual_stream scripts/run_wandb_sweep.slurm

set -euo pipefail

# Load environment
module load python/3.10  # Adjust to your cluster's Python module
module load cuda/11.8    # Adjust to your CUDA version

# Activate virtual environment
source .venv/bin/activate

# Setup directories
mkdir -p logs
mkdir -p training/results_wandb_sweep

# Wandb configuration
export WANDB_PROJECT="crop-futures-prediction"
export WANDB_ENTITY=""  # Set your wandb entity/username here
export WANDB_DIR="$PWD/wandb"

# Sweep ID - SET THIS after running: wandb sweep training/wandb_sweep.yaml
# Example: SWEEP_ID="crop-futures-prediction/abc123"
SWEEP_ID="${SWEEP_ID:-}"

# Number of runs per agent
RUNS_PER_AGENT="${RUNS_PER_AGENT:-5}"

echo "============================================================"
echo "Wandb Sweep Agent - Job $SLURM_ARRAY_TASK_ID"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Sweep ID: ${SWEEP_ID:-none (single run)}"
echo "============================================================"

# Check if sweep ID is set
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not set."
    echo "Initialize sweep first: wandb sweep training/wandb_sweep.yaml"
    echo "Then set SWEEP_ID in this script or use:"
    echo "  sbatch --export=SWEEP_ID=your/project/abc123 scripts/run_wandb_sweep.slurm"
    exit 1
fi

# Run wandb agent for this job
# Each agent will complete RUNS_PER_AGENT runs
if [ "$SWEEP_ID" = "none" ] || [ "$SWEEP_ID" = "single" ]; then
    # Run single experiment (for testing)
    echo "Running single experiment..."
    MODEL="${MODEL:-attention_vision_dual_stream}"
    COMMODITY="${COMMODITY:-corn}"
    HORIZON="${HORIZON:-1}"
    
    python3 training/train_wandb.py \
        --model "$MODEL" \
        --commodity "$COMMODITY" \
        --horizon "$HORIZON" \
        --epochs 100 \
        --no_early_stopping \
        --hidden_dim 128 \
        --num_layers 2 \
        --dropout 0.3 \
        --lr 0.001 \
        --seq_len 20 \
        --batch_size 32 \
        --output_dir training/results_wandb_sweep
else
    # Run sweep agent
    echo "Starting wandb agent for sweep: $SWEEP_ID"
    echo "This agent will run $RUNS_PER_AGENT configurations"
    
    # Count parameter ensures agent stops after N runs
    wandb agent --count "$RUNS_PER_AGENT" "$SWEEP_ID"
fi

echo "============================================================"
echo "Job completed at $(date)"
echo "============================================================"
