#!/bin/bash
#BSUB -J p1cell
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# One phase-1 cell. Set INDEX=1..24 or N+EXP in the submitting shell.
#   INDEX=17 bsub -J p1_17 < scripts/phase1_one.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT="/nfsshare/home/taotang/cursor/mage"
cd "${ROOT}"
exec bash "${ROOT}/scripts/phase1_cell.sh"
