#!/bin/bash
# Sequential benchmark batch — all 13M ablation models on RTX 3090
# Run: CUDA_VISIBLE_DEVICES=1 bash run_all_benchmarks.sh 2>&1 | tee benchmarks/logs/bench_batch_all.log
set -e
cd /home/dlewis3/Desktop/AI/DWARF
PY=".venv/bin/python3"

run_eval() {
    local arch=$1; local ckpt=$2; local tag=$3
    local ts=$(date +%Y%m%d_%H%M%S)
    echo ""
    echo "=========================================="
    echo "  $tag  [$arch]  $(date)"
    echo "=========================================="
    $PY -u benchmarks/eval_external.py \
        --arch "$arch" \
        --checkpoint "$ckpt" \
        2>&1 | tee "benchmarks/logs/eval_external_${tag}_${ts}.log"
    echo "  Done: $tag"
}

# condM 13M is already running — skip it (will have its own log)
# These run after it finishes:
run_eval condu_13m      checkpoints/condU/best.pt                          condU_13m
run_eval condm_13m      checkpoints/2048_condM_layer5_checkpoints/best.pt  condM_13m_L5
run_eval condm_13m_L0   checkpoints/2048_condM_layer0_checkpoints/best.pt  condM_13m_L0
run_eval condm_13m_L3   checkpoints/2048_condM_layer3_checkpoints/best.pt  condM_13m_L3
run_eval condm_13m      checkpoints/conds_gate0/best.pt                    condS_gate0
run_eval condm_13m      checkpoints/condq_bugfix/best.pt                   condQ_gate20

echo ""
echo "All batch benchmarks complete: $(date)"
