#!/bin/bash
#BSUB -J five_fit
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu02||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# Overfit 5 images with per-image condition ids, then judge token-level reproduction.
# Official conda env is torch 1.7.1 + CUDA 10.2, so V100 only.
#
# Success: outputs/five_fit/eval/metrics.json has perfect=true
#   every id0..id4 oneshot_token_acc == 1.0 (256/256)
# Pixel match is vs each image's vqgan_recon.png, not the raw input (VQGAN ceiling).

set -euo pipefail

export PYTHONNOUSERSITE=1

ROOT="/nfsshare/home/taotang/cursor/mage"
PYTHON="/nfsshare/home/taotang/miniforge3/envs/mage/bin/python"
TINY="${ROOT}/data/tiny-imagenet-200/train"
FIT="${ROOT}/data/five_fit"
OUT="${ROOT}/outputs/five_fit"
CLASSES=(n01443537 n01629819 n01641577 n01644900 n01698640)

cd "${ROOT}"
mkdir -p outputs/lsf "${OUT}/eval"
rm -f "${OUT}/log.txt" "${OUT}/checkpoint-last.pth"
rm -f "${OUT}"/events.out.tfevents.*
rm -rf "${FIT}/train"
mkdir -p "${FIT}/train"

for i in 0 1 2 3 4; do
    cls="${CLASSES[$i]}"
    mkdir -p "${FIT}/train/id${i}"
    cp -f "${TINY}/${cls}/images/${cls}_0.JPEG" "${FIT}/train/id${i}/img.JPEG"
    echo "id${i} ${cls} ${TINY}/${cls}/images/${cls}_0.JPEG"
done

echo "host=$(hostname) start=$(date)"
echo "python=${PYTHON}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== pretrain (5-image memorize, cond ids) ==="
"${PYTHON}" main_pretrain.py \
    --model mage_vit_base_patch16 \
    --batch_size 5 \
    --epochs 400 \
    --warmup_epochs 0 \
    --num_workers 0 \
    --max_samples 5 \
    --no_aug \
    --label_smoothing 0 \
    --dropout 0 \
    --weight_decay 0 \
    --lr 3e-3 \
    --min_lr 3e-3 \
    --refine_lr 3e-4 \
    --refine_after 0.9 \
    --mask_ratio_min 0 \
    --mask_ratio_max 1.0 \
    --fixed_mask_ratio 1.0 \
    --decoder_mask_token \
    --linear_head \
    --cond_ids \
    --no_amp \
    --cache_tokens \
    --repeat 32 \
    --save_freq 1 \
    --save_ckpt_freq 0 \
    --stop_token_acc 1.0 \
    --eval_freq 1 \
    --eval_num_iter 12 \
    --data_path "${FIT}" \
    --output_dir "${OUT}"

echo "=== eval (per-image token acc + VQGAN ceiling) ==="
set +e
"${PYTHON}" eval_five_fit.py \
    --ckpt "${OUT}/checkpoint-last.pth" \
    --data_path "${FIT}" \
    --model mage_vit_base_patch16 \
    --num_iter 12 \
    --output_dir "${OUT}/eval"
eval_rc=$?
set -e

echo "=== artifacts ==="
ls -lh "${OUT}/checkpoint-last.pth" || true
ls -lh "${OUT}/eval" || true
echo "=== last train logs ==="
tail -n 8 "${OUT}/log.txt" || true
echo "metrics:"
if [[ -f "${OUT}/eval/metrics.json" ]]; then
    cat "${OUT}/eval/metrics.json"
else
    echo "metrics.json missing"
fi
echo "done=$(date) eval_rc=${eval_rc}"
exit "${eval_rc}"
