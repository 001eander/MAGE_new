#!/bin/bash
#BSUB -J single_fit
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu02||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# Overfit 1 image, then judge token-level reproduction.
# Official conda env is torch 1.7.1 + CUDA 10.2, so V100 only.
#
# Success: outputs/single_fit/eval/metrics.json has
#   oneshot_token_acc == 1.0  (256/256 VQ codes; all-mask greedy generation)
# Pixel match is vs vqgan_recon.png, not the raw input (VQGAN ceiling).

set -euo pipefail

export PYTHONNOUSERSITE=1

ROOT="/nfsshare/home/taotang/cursor/mage"
PYTHON="/nfsshare/home/taotang/miniforge3/envs/mage/bin/python"
SRC="${SRC:-${ROOT}/data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG}"
FIT="${ROOT}/data/single_fit"
OUT="${ROOT}/outputs/single_fit"

cd "${ROOT}"
mkdir -p outputs/lsf "${FIT}/train/cls" "${OUT}/eval"
# Fresh run: do not resume the failed histogram-only fit.
rm -f "${OUT}/log.txt" "${OUT}/checkpoint-last.pth"
rm -f "${OUT}"/events.out.tfevents.*

cp -f "${SRC}" "${FIT}/train/cls/img.JPEG"

echo "host=$(hostname) start=$(date)"
echo "python=${PYTHON}"
echo "src=${SRC}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== pretrain (single-image memorize) ==="
"${PYTHON}" main_pretrain.py \
    --model mage_vit_base_patch16 \
    --batch_size 1 \
    --epochs 400 \
    --warmup_epochs 0 \
    --num_workers 0 \
    --max_samples 1 \
    --no_aug \
    --label_smoothing 0 \
    --dropout 0 \
    --weight_decay 0 \
    --lr 3e-3 \
    --min_lr 3e-3 \
    --mask_ratio_min 0 \
    --mask_ratio_max 1.0 \
    --fixed_mask_ratio 1.0 \
    --decoder_mask_token \
    --linear_head \
    --no_amp \
    --cache_tokens \
    --repeat 32 \
    --save_freq 20 \
    --save_ckpt_freq 0 \
    --stop_token_acc 1.0 \
    --eval_freq 1 \
    --eval_num_iter 12 \
    --data_path "${FIT}" \
    --output_dir "${OUT}"

echo "=== eval (token acc + VQGAN ceiling) ==="
set +e
"${PYTHON}" eval_single_fit.py \
    --ckpt "${OUT}/checkpoint-last.pth" \
    --image "${FIT}/train/cls/img.JPEG" \
    --model mage_vit_base_patch16 \
    --num_iter 12 \
    --output_dir "${OUT}/eval"
eval_rc=$?
set -e

echo "=== artifacts ==="
ls -lh "${OUT}/checkpoint-last.pth" || true
ls -lh "${OUT}/eval" || true
echo "=== last train logs ==="
tail -n 5 "${OUT}/log.txt" || true
echo "metrics:"
if [[ -f "${OUT}/eval/metrics.json" ]]; then
    cat "${OUT}/eval/metrics.json"
else
    echo "metrics.json missing"
fi
echo "done=$(date) eval_rc=${eval_rc}"
exit "${eval_rc}"
