#!/bin/bash
# Local Wandb Sweep Runner (no SLURM)
# 
# Usage:
#   1. Initialize sweep: ./scripts/run_wandb_sweep_local.sh init
#   2. Run agents: ./scripts/run_wandb_sweep_local.sh run SWEEP_ID NUM_AGENTS
#
# Example:
#   ./scripts/run_wandb_sweep_local.sh init
#   ./scripts/run_wandb_sweep_local.sh run crop-futures/abc123 4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Wandb configuration
export WANDB_PROJECT="crop-futures-prediction"
export WANDB_ENTITY=""  # Optional: set your wandb username/team
export WANDB_DIR="$PWD/wandb"

mkdir -p logs
mkdir -p training/results_wandb_sweep

COMMAND="${1:-help}"

if [ "$COMMAND" = "init" ]; then
    echo "Initializing wandb sweep..."
    wandb sweep training/wandb_sweep.yaml
    echo ""
    echo "Copy the sweep ID above and use it with: ./scripts/run_wandb_sweep_local.sh run SWEEP_ID NUM_AGENTS"
    
elif [ "$COMMAND" = "run" ]; then
    SWEEP_ID="${2:-}"
    NUM_AGENTS="${3:-4}"
    RUNS_PER_AGENT="${4:-5}"
    
    if [ -z "$SWEEP_ID" ]; then
        echo "Error: Sweep ID required"
        echo "Usage: $0 run SWEEP_ID [NUM_AGENTS] [RUNS_PER_AGENT]"
        echo "Example: $0 run crop-futures/abc123 4 5"
        exit 1
    fi
    
    echo "============================================================"
    echo "Running wandb sweep with $NUM_AGENTS parallel agents"
    echo "Sweep ID: $SWEEP_ID"
    echo "Runs per agent: $RUNS_PER_AGENT"
    echo "============================================================"
    
    # Run agents in parallel
    PIDS=()
    for i in $(seq 1 $NUM_AGENTS); do
        echo "Starting agent $i..."
        wandb agent --count "$RUNS_PER_AGENT" "$SWEEP_ID" > "logs/wandb_agent_$i.log" 2>&1 &
        PIDS+=($!)
    done
    
    echo ""
    echo "Agents started with PIDs: ${PIDS[@]}"
    echo "Logs: logs/wandb_agent_*.log"
    echo ""
    echo "To monitor progress:"
    echo "  - Check wandb dashboard: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
    echo "  - Watch logs: tail -f logs/wandb_agent_1.log"
    echo "  - Wait for completion: wait ${PIDS[@]}"
    echo ""
    
    # Wait for all agents
    wait ${PIDS[@]}
    
    echo "============================================================"
    echo "All agents completed!"
    echo "Check wandb dashboard for best hyperparameters"
    echo "============================================================"
    
elif [ "$COMMAND" = "single" ]; then
    # Run single experiment (useful for testing)
    echo "Running single experiment..."
    python3 training/train_wandb.py \
        --model attention_vision_dual_stream \
        --commodity corn \
        --horizon 1 \
        --epochs 100 \
        --no_early_stopping \
        --hidden_dim 128 \
        --num_layers 2 \
        --dropout 0.3 \
        --lr 0.001 \
        --seq_len 20 \
        --batch_size 32 \
        --output_dir training/results_wandb_sweep
        
elif [ "$COMMAND" = "grid" ]; then
    # Run manual grid search (without wandb)
    echo "Running manual grid search..."
    
    for hidden_dim in 64 128 256; do
        for dropout in 0.2 0.3; do
            for lr in 0.001 0.0005; do
                echo "Training: hidden=$hidden_dim, dropout=$dropout, lr=$lr"
                
                python3 training/train_wandb.py \
                    --model attention_vision_dual_stream \
                    --commodity corn \
                    --horizon 1 \
                    --epochs 50 \
                    --no_early_stopping \
                    --hidden_dim $hidden_dim \
                    --dropout $dropout \
                    --lr $lr \
                    --seq_len 20 \
                    --batch_size 32 \
                    --output_dir training/results_grid_search \
                    2>&1 | tee "logs/grid_h${hidden_dim}_d${dropout}_lr${lr}.log"
            done
        done
    done
    
    echo "Grid search complete! Check logs/ directory for results"
    
else
    echo "Wandb Sweep Helper Script"
    echo ""
    echo "Commands:"
    echo "  init                    - Initialize sweep and get sweep ID"
    echo "  run SWEEP_ID [N] [R]    - Run sweep with N agents, R runs each"
    echo "  single                  - Run single experiment (testing)"
    echo "  grid                    - Run manual grid search (no wandb)"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 run crop-futures/abc123 4 5"
    echo "  $0 single"
    echo "  $0 grid"
fi
