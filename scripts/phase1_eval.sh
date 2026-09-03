#!/bin/bash
# Eval-only for a cell that already has checkpoint-last.pth.
# Set INDEX=1..24 or N+EXP. Do not edit phase1_cell.sh while train jobs run.

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

if [[ ! -f "${CKPT}" ]]; then
    echo "missing ckpt ${CKPT}" >&2
    exit 1
fi

echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-} start=$(date) index=${INDEX:-} n=${N} exp=${EXP} ckpt=${CKPT}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

SCHEDULE="${SCHEDULE:-cosine}"
echo "=== eval ${EXP} n=${N} schedule=${SCHEDULE} ==="
eval_args=(
    --n "${N}"
    --exp_name "${EXP}"
    --output_dir "${OUT_DIR}"
    --ckpt "${CKPT}"
    --recon_dir "${RECON_DIR}"
    --schedule "${SCHEDULE}"
)
if [[ "${LINEAR_HEAD}" == "1" ]]; then
    eval_args+=(--linear_head)
fi
"${PYTHON}" eval_exp.py "${eval_args[@]}"
"${PYTHON}" scripts/make_phase1_table.py --n "${N}" --schedule "${SCHEDULE}" || true
echo "done=$(date) n=${N} exp=${EXP} schedule=${SCHEDULE}"
