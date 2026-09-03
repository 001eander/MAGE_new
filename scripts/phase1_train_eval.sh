#!/bin/bash
#BSUB -J phase1[1-16]
#BSUB -oo outputs/lsf/%J_%I.out
#BSUB -eo outputs/lsf/%J_%I.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# Fallback single-cell array. Preferred path is scripts/phase1_fill_pack.py
# packing several cells onto one multi-GPU job.

set -euo pipefail
ROOT="/nfsshare/home/taotang/cursor/mage"
cd "${ROOT}"
if [[ -n "${LSB_JOBINDEX:-}" ]]; then
    export INDEX="${LSB_JOBINDEX}"
fi
exec bash "${ROOT}/scripts/phase1_cell.sh"
