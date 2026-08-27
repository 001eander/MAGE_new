#!/bin/bash
#BSUB -J eval_mage
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu02||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# Unified eval. Defaults match splits/manifest.json.
#   N=2 EXP=dot_smooth CKPT=outputs/dot_smooth/n2/checkpoint-last.pth bsub < scripts/eval_exp.sh
# Recon FID only (same script / same FID function):
#   N=2 EXP=_ref RECON_FID_ONLY=1 bsub < scripts/eval_exp.sh

set -euo pipefail
export PYTHONNOUSERSITE=1

ROOT="/nfsshare/home/taotang/cursor/mage"
PYTHON="/nfsshare/home/taotang/miniforge3/envs/mage/bin/python"

N="${N:-2}"
EXP="${EXP:-exp}"
CKPT="${CKPT:-}"
RECON_FID_ONLY="${RECON_FID_ONLY:-0}"
GEN_DIR="${GEN_DIR:-}"

cd "${ROOT}"
mkdir -p outputs/lsf

echo "host=$(hostname) start=$(date) n=${N} exp=${EXP}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

args=(
    --n "${N}"
    --exp_name "${EXP}"
    --output_dir "outputs/${EXP}/n${N}"
)
if [[ "${RECON_FID_ONLY}" == "1" ]]; then
    args+=(--recon_fid_only)
else
    if [[ -n "${CKPT}" ]]; then
        args+=(--ckpt "${CKPT}")
    fi
    if [[ -n "${GEN_DIR}" ]]; then
        args+=(--gen_dir "${GEN_DIR}")
    fi
fi

"${PYTHON}" eval_exp.py "${args[@]}"
echo "done=$(date)"
