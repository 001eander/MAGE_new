#!/bin/bash
#BSUB -J smoke_mage
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu02||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# Single-GPU smoke: 1 image, 1 pretrain epoch, 1 generated image.
# Official conda env is torch 1.7.1 + CUDA 10.2, so V100 only.

set -euo pipefail

# Keep ~/.local site-packages from shadowing the conda env (e.g. numpy).
export PYTHONNOUSERSITE=1

ROOT="/nfsshare/home/taotang/cursor/mage"
PYTHON="/nfsshare/home/taotang/miniforge3/envs/mage/bin/python"
SRC="${ROOT}/data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG"
SMOKE="${ROOT}/data/smoke"
OUT="${ROOT}/outputs/smoke"

cd "${ROOT}"
mkdir -p outputs/lsf "${SMOKE}/train/cls" "${OUT}"

cp -f "${SRC}" "${SMOKE}/train/cls/img.JPEG"

echo "host=$(hostname) start=$(date)"
echo "python=${PYTHON}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== pretrain ==="
"${PYTHON}" main_pretrain.py \
    --model mage_vit_base_patch16 \
    --batch_size 1 \
    --epochs 1 \
    --warmup_epochs 0 \
    --num_workers 0 \
    --max_samples 1 \
    --data_path "${SMOKE}" \
    --output_dir "${OUT}" \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --mask_ratio_min 0.5 \
    --mask_ratio_max 1.0 \
    --mask_ratio_mu 0.55 \
    --mask_ratio_std 0.25

echo "=== generate ==="
"${PYTHON}" gen_img_uncond.py \
    --ckpt "${OUT}/checkpoint-last.pth" \
    --model mage_vit_base_patch16 \
    --batch_size 1 \
    --num_images 1 \
    --num_iter 4 \
    --temp 6.0 \
    --output_dir "${OUT}/gen"

echo "=== artifacts ==="
ls -lh "${OUT}/checkpoint-last.pth"
ls -lh "${OUT}/gen"/temp*
echo "done=$(date)"
