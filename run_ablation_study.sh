#!/bin/bash

################################################################################
# Ablation Study for RDLGC with BYOL + Prototype Learning
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_experiment() {
    echo -e "${BLUE}[EXPERIMENT]${NC} $1"
}

################################################################################
# Configuration
################################################################################

DATASET=${1:-"mvtec"}  # mvtec or visa
GPU=${2:-"0"}          # GPU ID
MODE=${3:-"all"}       # all, dense, global, prototype, module_combination

print_header "Ablation Study: RDLGC BYOL + Prototype"
print_info "Dataset: $DATASET"
print_info "GPU: $GPU"
print_info "Study Type: $MODE"

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/ablation

CONFIG="configs/rd/rd_byol_proto_${DATASET}.py"

################################################################################
# Module 1: DenseLoss Ablation
################################################################################

ablation_dense_loss() {
    print_header "Ablation Study: DenseLoss Module"

    # Experiment 1: No DenseLoss
    print_experiment "1. Baseline WITHOUT DenseLoss"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.1.lam=0.0 \
        trainer.logdir_sub=ablation/dense_disabled \
        2>&1 | tee logs/ablation/dense_disabled.log

    # Experiment 2: DenseLoss with different weights
    for lam_dense in 0.5 1.0 2.0; do
        print_experiment "2. DenseLoss weight = $lam_dense"
        python run.py \
            -c $CONFIG \
            loss.loss_terms.1.lam=$lam_dense \
            trainer.logdir_sub=ablation/dense_lam_${lam_dense} \
            2>&1 | tee logs/ablation/dense_lam_${lam_dense}.log
    done

    # Experiment 3: DenseLoss with/without spatial matching
    print_experiment "3. DenseLoss WITHOUT spatial matching"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.1.use_spatial_matching=False \
        trainer.logdir_sub=ablation/dense_no_spatial \
        2>&1 | tee logs/ablation/dense_no_spatial.log
}

################################################################################
# Module 2: GlobalLoss (BYOL) Ablation
################################################################################

ablation_global_loss() {
    print_header "Ablation Study: GlobalLoss (BYOL) Module"

    # Experiment 1: No Global BYOL Loss (only Prototype)
    print_experiment "1. WITHOUT BYOL Global Loss (Prototype only)"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.2.lam=0.0 \
        trainer.logdir_sub=ablation/global_disabled \
        2>&1 | tee logs/ablation/global_disabled.log

    # Experiment 2: Global BYOL with different weights
    for lam_global in 0.5 1.0 2.0; do
        print_experiment "2. Global BYOL weight = $lam_global"
        python run.py \
            -c $CONFIG \
            loss.loss_terms.2.lam=$lam_global \
            trainer.logdir_sub=ablation/global_lam_${lam_global} \
            2>&1 | tee logs/ablation/global_lam_${lam_global}.log
    done
}

################################################################################
# Module 3: Prototype Learning Ablation
################################################################################

ablation_prototype() {
    print_header "Ablation Study: Prototype Learning Module"

    # Experiment 1: No Prototype Learning (only BYOL)
    print_experiment "1. WITHOUT Prototype Learning (BYOL only)"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.2.lam_proto=0.0 \
        trainer.logdir_sub=ablation/proto_disabled \
        2>&1 | tee logs/ablation/proto_disabled.log

    # Experiment 2: Prototype weight ablation
    print_info "=== Prototype Weight Ablation ==="
    for lam_proto in 0.5 1.0 1.5 2.0; do
        print_experiment "2. Prototype weight = $lam_proto"
        python run.py \
            -c $CONFIG \
            loss.loss_terms.2.lam_proto=$lam_proto \
            trainer.logdir_sub=ablation/proto_lam_${lam_proto} \
            2>&1 | tee logs/ablation/proto_lam_${lam_proto}.log
    done

    # Experiment 3: Number of prototypes ablation
    print_info "=== Number of Prototypes Ablation ==="
    for n_proto in 3 5 7 10; do
        print_experiment "3. Number of prototypes = $n_proto"
        python run.py \
            -c $CONFIG \
            loss.loss_terms.2.n_prototypes=$n_proto \
            trainer.logdir_sub=ablation/proto_n_${n_proto} \
            2>&1 | tee logs/ablation/proto_n_${n_proto}.log
    done

    # Experiment 4: Temperature ablation
    print_info "=== Temperature Ablation ==="
    for temp in 0.05 0.07 0.10 0.15; do
        print_experiment "4. Temperature = $temp"
        python run.py \
            -c $CONFIG \
            loss.loss_terms.2.temperature=$temp \
            trainer.logdir_sub=ablation/proto_temp_${temp} \
            2>&1 | tee logs/ablation/proto_temp_${temp}.log
    done
}

################################################################################
# Module Combination Ablation
################################################################################

ablation_module_combination() {
    print_header "Ablation Study: Module Combinations"

    # Full baseline (all modules)
    print_experiment "0. Full Model (All Modules Enabled)"
    python run.py \
        -c $CONFIG \
        trainer.logdir_sub=ablation/full_model \
        2>&1 | tee logs/ablation/full_model.log

    # Only CosLoss (minimal baseline)
    print_experiment "1. Only CosLoss"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.1.lam=0.0 \
        loss.loss_terms.2.lam=0.0 \
        loss.loss_terms.2.lam_proto=0.0 \
        trainer.logdir_sub=ablation/only_cos \
        2>&1 | tee logs/ablation/only_cos.log

    # CosLoss + DenseLoss
    print_experiment "2. CosLoss + DenseLoss"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.2.lam=0.0 \
        loss.loss_terms.2.lam_proto=0.0 \
        trainer.logdir_sub=ablation/cos_dense \
        2>&1 | tee logs/ablation/cos_dense.log

    # CosLoss + GlobalLoss (BYOL)
    print_experiment "3. CosLoss + GlobalLoss (BYOL only)"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.1.lam=0.0 \
        loss.loss_terms.2.lam_proto=0.0 \
        trainer.logdir_sub=ablation/cos_global \
        2>&1 | tee logs/ablation/cos_global.log

    # CosLoss + Prototype
    print_experiment "4. CosLoss + Prototype (no dense, no BYOL)"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.1.lam=0.0 \
        loss.loss_terms.2.lam=0.0 \
        trainer.logdir_sub=ablation/cos_proto \
        2>&1 | tee logs/ablation/cos_proto.log

    # DenseLoss + GlobalLoss
    print_experiment "5. DenseLoss + GlobalLoss (no prototype)"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.2.lam_proto=0.0 \
        trainer.logdir_sub=ablation/dense_global \
        2>&1 | tee logs/ablation/dense_global.log

    # DenseLoss + Prototype
    print_experiment "6. DenseLoss + Prototype (no global BYOL)"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.2.lam=0.0 \
        trainer.logdir_sub=ablation/dense_proto \
        2>&1 | tee logs/ablation/dense_proto.log

    # GlobalLoss + Prototype
    print_experiment "7. GlobalLoss + Prototype (no dense)"
    python run.py \
        -c $CONFIG \
        loss.loss_terms.1.lam=0.0 \
        trainer.logdir_sub=ablation/global_proto \
        2>&1 | tee logs/ablation/global_proto.log
}

################################################################################
# Main Execution
################################################################################

case $MODE in
    dense)
        ablation_dense_loss
        ;;

    global)
        ablation_global_loss
        ;;

    prototype)
        ablation_prototype
        ;;

    module_combination)
        ablation_module_combination
        ;;

    all)
        print_header "Running ALL Ablation Studies"
        ablation_dense_loss
        ablation_global_loss
        ablation_prototype
        ablation_module_combination
        ;;

    *)
        print_error "Unknown mode: $MODE"
        print_info "Usage: $0 [DATASET] [GPU] [MODE]"
        print_info ""
        print_info "DATASET: mvtec, visa (default: mvtec)"
        print_info "GPU: GPU ID (default: 0)"
        print_info "MODE:"
        print_info "  dense              - Ablation for DenseLoss module"
        print_info "  global             - Ablation for GlobalLoss (BYOL) module"
        print_info "  prototype          - Ablation for Prototype Learning module"
        print_info "  module_combination - Ablation for different module combinations"
        print_info "  all                - Run all ablation studies (default)"
        print_info ""
        print_info "Examples:"
        print_info "  $0 mvtec 0 dense              # Test DenseLoss on MVTec"
        print_info "  $0 mvtec 0 prototype          # Test Prototype on MVTec"
        print_info "  $0 mvtec 0 module_combination # Test module combinations"
        print_info "  $0 mvtec 0 all                # Run all ablations"
        exit 1
        ;;
esac

print_header "Ablation Study Complete!"
