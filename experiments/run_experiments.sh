#!/bin/bash
# Run experiments for MedEyes

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
DATA_ROOT="./data"
OUTPUT_DIR="./outputs/experiments"
CHECKPOINT_DIR="./checkpoints"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

# Function to run single experiment
run_experiment() {
    local dataset=$1
    local config=$2
    local name=$3

    echo "=========================================="
    echo "Running experiment: $name"
    echo "Dataset: $dataset"
    echo "Config: $config"
    echo "=========================================="

    # Training
    python scripts/train.py \
        --config $config \
        --dataset $dataset \
        --output_dir "$OUTPUT_DIR/$name" \
        --distributed \
        2>&1 | tee "$OUTPUT_DIR/$name/train.log"

    # Evaluation
    python scripts/evaluate.py \
        --checkpoint "$OUTPUT_DIR/$name/checkpoint_best.pth" \
        --dataset $dataset \
        --split test \
        --save_predictions \
        --output_dir "$OUTPUT_DIR/$name/eval" \
        2>&1 | tee "$OUTPUT_DIR/$name/eval.log"
}

# Main experiments
echo "Starting MedEyes experiments..."

# Experiment 1: VQA-RAD
run_experiment "vqa-rad" "configs/default.yaml" "medeyes_vqa_rad"

# Experiment 2: SLAKE
run_experiment "slake" "configs/default.yaml" "medeyes_slake"

# Experiment 3: PathVQA
run_experiment "pathvqa" "configs/default.yaml" "medeyes_pathvqa"

# Ablation studies
echo "Running ablation studies..."

# Ablation 1: Without GRN
python scripts/train.py \
    --config configs/default.yaml \
    --dataset vqa-rad \
    --output_dir "$OUTPUT_DIR/ablation_no_grn" \
    --no_grn \
    2>&1 | tee "$OUTPUT_DIR/ablation_no_grn/train.log"

# Ablation 2: Without CVS
python scripts/train.py \
    --config configs/default.yaml \
    --dataset vqa-rad \
    --output_dir "$OUTPUT_DIR/ablation_no_cvs" \
    --no_cvs \
    2>&1 | tee "$OUTPUT_DIR/ablation_no_cvs/train.log"

# Ablation 3: Without off-policy
python scripts/train.py \
    --config configs/default.yaml \
    --dataset vqa-rad \
    --output_dir "$OUTPUT_DIR/ablation_no_offpolicy" \
    --no_offpolicy \
    2>&1 | tee "$OUTPUT_DIR/ablation_no_offpolicy/train.log"

# Generate final report
python scripts/generate_report.py \
    --experiment_dir $OUTPUT_DIR \
    --output "$OUTPUT_DIR/final_report.html"

echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"