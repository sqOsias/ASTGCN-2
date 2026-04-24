#!/bin/bash
# ============================================================
# Baseline Models — Train & Test Script (LSTM & GRU)
# Dataset: PEMS04 | Horizons: 3 and 12 (aligned with ASTGCN)
#
# Usage:
#   cd /root/ASTGCN-2
#
#   # --- Train (saves checkpoint + training log + meta) ---
#   bash baselines/train_baselines.sh train           # train both models, h=3 & h=12
#   bash baselines/train_baselines.sh train lstm      # train LSTM only, h=3 & h=12
#   bash baselines/train_baselines.sh train gru       # train GRU only, h=3 & h=12
#
#   # --- Test (loads checkpoint, outputs metrics + predictions) ---
#   bash baselines/train_baselines.sh test            # test both models, h=3 & h=12
#   bash baselines/train_baselines.sh test lstm       # test LSTM only
#   bash baselines/train_baselines.sh test gru        # test GRU only
#
#   # --- Train + Test together ---
#   bash baselines/train_baselines.sh all             # full pipeline
# ============================================================

set -e
cd "$(dirname "$0")/.."   # ensure we are in project root

# ---- Hyperparameters (modify as needed) ----
DATA_PATH="data/PEMS04/pems04.npz"
SENSOR_ID=0
HIDDEN_SIZE=64
BATCH_SIZE=64
LR=0.001
EPOCHS=100
SEQ_LEN=12
HORIZONS="3 12"     # aligned with ASTGCN

PHASE=${1:-all}
MODEL=${2:-both}

UNIFIED_CSV="baselines/results/all_test_metrics.csv"

# ======================== TRAIN ========================
train_model() {
    local model_name=$1   # lstm or gru
    local module="baselines.run_${model_name}"
    for H in $HORIZONS; do
        echo "============================================================"
        echo "  [TRAIN] ${model_name^^} Baseline  horizon=${H}"
        echo "============================================================"
        python -m "$module" \
            --data_path "$DATA_PATH" \
            --sensor_id "$SENSOR_ID" \
            --hidden_size "$HIDDEN_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --seq_len "$SEQ_LEN" \
            --horizon "$H"
        echo ""
    done
}

# ======================== TEST =========================
test_model() {
    local model_name=$1   # lstm or gru
    local module="baselines.test_${model_name}"
    for H in $HORIZONS; do
        echo "============================================================"
        echo "  [TEST] ${model_name^^} Baseline  horizon=${H}"
        echo "============================================================"
        python -m "$module" --batch_size "$BATCH_SIZE" --horizon "$H"
        echo ""
    done
}

# ======================== DISPATCH =====================
do_train() {
    case "$MODEL" in
        lstm) train_model lstm ;;
        gru)  train_model gru  ;;
        both) train_model lstm; train_model gru ;;
        *)    echo "Unknown model: $MODEL (use lstm|gru|both)"; exit 1 ;;
    esac
}

do_test() {
    # Clear unified CSV so we get a fresh file per run
    rm -f "$UNIFIED_CSV"
    case "$MODEL" in
        lstm) test_model lstm ;;
        gru)  test_model gru  ;;
        both) test_model lstm; test_model gru ;;
        *)    echo "Unknown model: $MODEL (use lstm|gru|both)"; exit 1 ;;
    esac
}

case "$PHASE" in
    train) do_train ;;
    test)  do_test  ;;
    all)   do_train; do_test ;;
    *)     echo "Usage: $0 [train|test|all] [lstm|gru|both]"; exit 1 ;;
esac

# ======================== SUMMARY ======================
echo "============================================================"
echo "  All test metrics (unified):"
echo "============================================================"
if [ -f "$UNIFIED_CSV" ]; then
    cat "$UNIFIED_CSV"
else
    echo "  (no test results yet — run with 'test' or 'all')"
fi
echo ""
