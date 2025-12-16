#!/bin/bash

################################################################################
# Quick Start Script for RDLGC with Prototype Learning
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

################################################################################
# Configuration
################################################################################

DATASET=${1:-"mvtec"}  # mvtec or visa
MODE=${2:-"train"}     # train, test, or ablation
GPU=${3:-"0"}          # GPU ID
DATA_PATH=${DATA_PATH:-""}  # Optional: set via environment variable or modify here

print_header "RDLGC with Prototype Learning"
print_info "Dataset: $DATASET"
print_info "Mode: $MODE"
print_info "GPU: $GPU"
if [ -n "$DATA_PATH" ]; then
    print_info "Data Path: $DATA_PATH"
fi

################################################################################
# Set CUDA device
################################################################################

export CUDA_VISIBLE_DEVICES=$GPU

################################################################################
# Training Functions
################################################################################

train_mvtec() {
    print_header "Training on MVTec with Prototypes"
    mkdir -p logs
    DATA_PATH_ARG=""
    if [ -n "$DATA_PATH" ]; then
        DATA_PATH_ARG="--data_path $DATA_PATH"
    fi
    python run.py \
        -c configs/rd/rd_byol_proto_mvtec.py \
        $DATA_PATH_ARG \
        2>&1 | tee logs/mvtec_proto_$(date +%Y%m%d_%H%M%S).log
}

train_visa() {
    print_header "Training on VisA with Prototypes"
    mkdir -p logs
    DATA_PATH_ARG=""
    if [ -n "$DATA_PATH" ]; then
        DATA_PATH_ARG="--data_path $DATA_PATH"
    fi
    python run.py \
        -c configs/rd/rd_byol_proto_visa.py \
        $DATA_PATH_ARG \
        2>&1 | tee logs/visa_proto_$(date +%Y%m%d_%H%M%S).log
}

################################################################################
# Testing Functions
################################################################################

test_mvtec() {
    print_header "Testing on MVTec"
    CHECKPOINT_DIR=${4:-""}
    if [ -z "$CHECKPOINT_DIR" ]; then
        print_error "Please provide checkpoint directory"
        exit 1
    fi

    DATA_PATH_ARG=""
    if [ -n "$DATA_PATH" ]; then
        DATA_PATH_ARG="--data_path $DATA_PATH"
    fi
    python run.py \
        -c configs/rd/rd_byol_proto_mvtec.py \
        -m test \
        $DATA_PATH_ARG \
        resume_dir="$CHECKPOINT_DIR"
}

test_visa() {
    print_header "Testing on VisA"
    CHECKPOINT_DIR=${4:-""}
    if [ -z "$CHECKPOINT_DIR" ]; then
        print_error "Please provide checkpoint directory"
        exit 1
    fi

    DATA_PATH_ARG=""
    if [ -n "$DATA_PATH" ]; then
        DATA_PATH_ARG="--data_path $DATA_PATH"
    fi
    python run.py \
        -c configs/rd/rd_byol_proto_visa.py \
        -m test \
        $DATA_PATH_ARG \
        resume_dir="$CHECKPOINT_DIR"
}

################################################################################
# Ablation Study Functions
################################################################################

ablation_loss_weights() {
    print_header "Ablation: Loss Weights"

    DATA_PATH_ARG=""
    if [ -n "$DATA_PATH" ]; then
        DATA_PATH_ARG="--data_path $DATA_PATH"
    fi
    mkdir -p logs
    for lam_proto in 0.5 1.0 1.5 2.0; do
        print_info "Testing lam_proto=$lam_proto"
        python run.py \
            -c configs/rd/rd_byol_proto_mvtec.py \
            $DATA_PATH_ARG \
            loss_terms.scl.lam_proto=$lam_proto \
            trainer.logdir_sub=ablation_lam_${lam_proto} \
            2>&1 | tee logs/ablation_lam_${lam_proto}.log
    done
}

ablation_n_prototypes() {
    print_header "Ablation: Number of Prototypes"

    DATA_PATH_ARG=""
    if [ -n "$DATA_PATH" ]; then
        DATA_PATH_ARG="--data_path $DATA_PATH"
    fi
    mkdir -p logs
    for n_proto in 3 5 7 10; do
        print_info "Testing n_prototypes=$n_proto"
        python run.py \
            -c configs/rd/rd_byol_proto_mvtec.py \
            $DATA_PATH_ARG \
            loss_terms.scl.n_prototypes=$n_proto \
            trainer.logdir_sub=ablation_proto_${n_proto} \
            2>&1 | tee logs/ablation_proto_${n_proto}.log
    done
}

ablation_temperature() {
    print_header "Ablation: Temperature"

    mkdir -p logs
    for temp in 0.05 0.07 0.10 0.15; do
        print_info "Testing temperature=$temp"
        python run.py \
            -c configs/rd/rd_byol_proto_mvtec.py \
            loss_terms.scl.temperature=$temp \
            trainer.logdir_sub=ablation_temp_${temp} \
            2>&1 | tee logs/ablation_temp_${temp}.log
    done
}

ablation_all() {
    print_header "Running All Ablation Studies"
    ablation_loss_weights
    ablation_n_prototypes
    ablation_temperature
}

################################################################################
# Comparison with Baseline
################################################################################

compare_baseline() {
    print_header "Comparing with Baseline RDLGC"

    mkdir -p logs
    # Train baseline
    print_info "Training baseline (without prototypes)"
    python run.py \
        -c configs/rd/rd_byol_mvtec.py \
        trainer.logdir_sub=baseline \
        2>&1 | tee logs/baseline_$(date +%Y%m%d_%H%M%S).log

    # Train with prototypes
    print_info "Training with prototypes"
    python run.py \
        -c configs/rd/rd_byol_proto_mvtec.py \
        trainer.logdir_sub=prototype \
        2>&1 | tee logs/prototype_$(date +%Y%m%d_%H%M%S).log
}

################################################################################
# Multi-GPU Training
################################################################################

train_multi_gpu() {
    NUM_GPUS=${4:-4}
    print_header "Multi-GPU Training ($NUM_GPUS GPUs)"

    mkdir -p logs
    CONFIG="configs/rd/rd_byol_proto_${DATASET}.py"

    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        run.py \
        -c $CONFIG \
        2>&1 | tee logs/${DATASET}_multi_gpu_$(date +%Y%m%d_%H%M%S).log
}

################################################################################
# Main Execution
################################################################################

case $MODE in
    train)
        if [ "$DATASET" = "mvtec" ]; then
            train_mvtec
        elif [ "$DATASET" = "visa" ]; then
            train_visa
        else
            print_error "Unknown dataset: $DATASET"
            exit 1
        fi
        ;;

    test)
        if [ "$DATASET" = "mvtec" ]; then
            test_mvtec
        elif [ "$DATASET" = "visa" ]; then
            test_visa
        else
            print_error "Unknown dataset: $DATASET"
            exit 1
        fi
        ;;

    ablation)
        ABLATION_TYPE=${4:-"all"}
        case $ABLATION_TYPE in
            weights)
                ablation_loss_weights
                ;;
            prototypes)
                ablation_n_prototypes
                ;;
            temperature)
                ablation_temperature
                ;;
            all)
                ablation_all
                ;;
            *)
                print_error "Unknown ablation type: $ABLATION_TYPE"
                print_info "Available: weights, prototypes, temperature, all"
                exit 1
                ;;
        esac
        ;;

    compare)
        compare_baseline
        ;;

    multi_gpu)
        train_multi_gpu
        ;;

    *)
        print_error "Unknown mode: $MODE"
        print_info "Usage: $0 [DATASET] [MODE] [GPU] [EXTRA_ARGS]"
        print_info ""
        print_info "DATASET: mvtec, visa (default: mvtec)"
        print_info "MODE:"
        print_info "  train       - Train model"
        print_info "  test        - Test model (requires checkpoint dir as 4th arg)"
        print_info "  ablation    - Run ablation studies (specify type as 4th arg)"
        print_info "  compare     - Compare with baseline"
        print_info "  multi_gpu   - Multi-GPU training (specify num_gpus as 4th arg)"
        print_info "GPU: GPU ID (default: 0)"
        print_info ""
        print_info "Examples:"
        print_info "  $0 mvtec train 0                    # Train on MVTec using GPU 0"
        print_info "  $0 visa train 1                     # Train on VisA using GPU 1"
        print_info "  $0 mvtec test 0 path/to/checkpoint  # Test on MVTec"
        print_info "  $0 mvtec ablation 0 weights         # Ablation on loss weights"
        print_info "  $0 mvtec ablation 0 all             # All ablations"
        print_info "  $0 mvtec compare 0                  # Compare with baseline"
        print_info "  $0 mvtec multi_gpu 0 4              # Train on 4 GPUs"
        exit 1
        ;;
esac

print_header "Done!"
