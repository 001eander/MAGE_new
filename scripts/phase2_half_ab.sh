#!/bin/bash
#BSUB -J p2_ab
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# Phase 2: half-A half-B stitching, one (n, experiment) cell.
#   N=10 EXP=lin_nosmooth bsub < scripts/phase2_half_ab.sh
# Optional: PATTERNS=leftright,checker SCHEDULES=oneshot,uncover1 MAX_PAIRS=16
# Output: outputs/<EXP>/n<N>/phase2_half_ab/<pattern>/metrics.json + grid_*.png

set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT="/nfsshare/home/taotang/GenModels/MAGE"
PYTHON="/nfsshare/home/taotang/miniforge3/envs/mage/bin/python"
cd "${ROOT}"
mkdir -p outputs/lsf

N="${N:-2}"
EXP="${EXP:-lin_nosmooth}"
PATTERNS="${PATTERNS:-leftright,checker}"
SCHEDULES="${SCHEDULES:-oneshot,uncover1}"
MAX_PAIRS="${MAX_PAIRS:-16}"

echo "host=$(hostname) n=${N} exp=${EXP} patterns=${PATTERNS} schedules=${SCHEDULES}"
for pattern in ${PATTERNS//,/ }; do
    echo "pattern ${pattern} start=$(date)"
    "${PYTHON}" eval_half_ab.py \
        --n "${N}" \
        --exp_name "${EXP}" \
        --pattern "${pattern}" \
        --schedules "${SCHEDULES}" \
        --max_pairs "${MAX_PAIRS}" \
        --device cuda
done
