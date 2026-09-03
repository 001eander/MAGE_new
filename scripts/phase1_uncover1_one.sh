#!/bin/bash
#BSUB -J p1u
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# Eval-only, one-position-at-a-time generation. Set INDEX=1..16.

set -euo pipefail
export SCHEDULE=uncover1
ROOT="/nfsshare/home/taotang/cursor/mage"
cd "${ROOT}"
exec bash "${ROOT}/scripts/phase1_eval.sh"
