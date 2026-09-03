#!/bin/bash
#BSUB -J phase1_prep
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu02||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# Shared Tiny val256 + VQGAN recon used by every phase-1 eval.

set -euo pipefail
export PYTHONNOUSERSITE=1

ROOT="/nfsshare/home/taotang/cursor/mage"
PYTHON="/nfsshare/home/taotang/miniforge3/envs/mage/bin/python"

cd "${ROOT}"
mkdir -p outputs/lsf outputs/_ref

echo "host=$(hostname) start=$(date)"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

n_png=$(ls data/tiny-imagenet-val256/*.png 2>/dev/null | wc -l)
if [[ "${n_png}" -lt 10000 ]]; then
    echo "preparing Tiny val256 (${n_png} pngs so far)"
    "${PYTHON}" scripts/prepare_tiny_val256.py
else
    echo "tiny-imagenet-val256 already has ${n_png} pngs"
fi

"${PYTHON}" eval_exp.py \
    --n 1 \
    --exp_name _ref \
    --output_dir outputs/_ref \
    --recon_dir outputs/_ref/tiny_val256_vqgan_recon \
    --recon_fid_only

echo "done=$(date)"
