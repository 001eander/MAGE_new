#!/bin/bash
# One phase-1 cell. No LSF header — called by phase1_one.sh or a pack worker.
# Set INDEX=1..24 or N+EXP in the environment.

set -euo pipefail
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ROOT="/nfsshare/home/taotang/cursor/mage"
PYTHON="/nfsshare/home/taotang/miniforge3/envs/mage/bin/python"

cd "${ROOT}"
mkdir -p outputs/lsf

if [[ -n "${INDEX:-}" ]]; then
    eval "$("${PYTHON}" -c "from util.phase1 import shell_exports; print(shell_exports(index=${INDEX}))")"
elif [[ -n "${N:-}" && -n "${EXP:-}" ]]; then
    eval "$("${PYTHON}" -c "from util.phase1 import shell_exports; print(shell_exports(n=${N}, exp_name='${EXP}'))")"
else
    echo "need INDEX or N+EXP" >&2
    exit 1
fi

echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-} start=$(date) index=${INDEX:-} n=${N} exp=${EXP} linear_head=${LINEAR_HEAD} ls=${LABEL_SMOOTHING}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

train_args=(
    --model mage_vit_base_patch16
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --lr "${LR}"
    --min_lr "${MIN_LR}"
    --weight_decay 0.05
    --num_workers 0
    --no_aug
    --image_list "${IMAGE_LIST}"
    --data_path data/tiny-imagenet-200
    --output_dir "${OUT_DIR}"
    --label_smoothing "${LABEL_SMOOTHING}"
    --mask_ratio_min 0.5
    --mask_ratio_max 1.0
    --mask_ratio_mu 0.55
    --mask_ratio_std 0.25
    --save_last_freq "${SAVE_LAST_FREQ}"
    --save_ckpt_freq 0
    --log_tb_freq "${LOG_TB_FREQ}"
    --seed 0
)
if [[ "${LINEAR_HEAD}" == "1" ]]; then
    train_args+=(--linear_head)
fi
if [[ "${CACHE_TOKENS}" == "1" ]]; then
    train_args+=(--cache_tokens)
fi
if [[ "${PROFILE}" == "1" ]]; then
    train_args+=(--profile)
fi
if [[ -f "${CKPT}" ]]; then
    train_args+=(--resume "${CKPT}")
    echo "resume ${CKPT}"
fi

echo "=== train ${EXP} n=${N} ==="
"${PYTHON}" main_pretrain.py "${train_args[@]}"

echo "=== eval ${EXP} n=${N} ==="
eval_args=(
    --n "${N}"
    --exp_name "${EXP}"
    --output_dir "${OUT_DIR}"
    --ckpt "${CKPT}"
    --recon_dir "${RECON_DIR}"
)
if [[ "${LINEAR_HEAD}" == "1" ]]; then
    eval_args+=(--linear_head)
fi
"${PYTHON}" eval_exp.py "${eval_args[@]}"

"${PYTHON}" scripts/make_phase1_table.py --n "${N}" || true
echo "done=$(date) n=${N} exp=${EXP}"
