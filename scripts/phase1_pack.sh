#!/bin/bash
#BSUB -J p1pack
#BSUB -oo outputs/lsf/%J.out
#BSUB -eo outputs/lsf/%J.err
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "order[ut]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[hname==gpu01||hname==gpu03||hname==gpu04||hname==gpu05||hname==gpu06||hname==gpu07||hname==gpu08||hname==gpu11||hname==gpu12]"

# One LSF job, N cards, N independent cells.
# Submitter overrides -gpu / -J / host select and sets PACK_INDICES=1,2,...
# This cluster leaves CUDA_VISIBLE_DEVICES as the allocated physical IDs
# (e.g. 1,3). Each worker takes the i-th ID. Do not rewrite them to 0..k-1.

set -uo pipefail
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ROOT="/nfsshare/home/taotang/cursor/mage"
cd "${ROOT}"
mkdir -p outputs/lsf

if [[ -z "${PACK_INDICES:-}" ]]; then
    echo "need PACK_INDICES=1,2,..." >&2
    exit 1
fi

IFS=',' read -ra IDXS <<< "${PACK_INDICES}"
parent_cuda="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${parent_cuda}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "${parent_cuda}"
else
    GPU_IDS=()
    for ((j = 0; j < ${#IDXS[@]}; j++)); do
        GPU_IDS+=("${j}")
    done
fi
if (( ${#GPU_IDS[@]} < ${#IDXS[@]} )); then
    echo "need ${#IDXS[@]} GPUs, CUDA_VISIBLE_DEVICES=${parent_cuda}" >&2
    exit 1
fi
echo "host=$(hostname) job=${LSB_JOBID:-} ngpu=${#IDXS[@]} indices=${PACK_INDICES} start=$(date) cuda_parent=${parent_cuda} gpu_ids=${GPU_IDS[*]}"

pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' TERM INT

for i in "${!IDXS[@]}"; do
    idx="${IDXS[$i]}"
    gpu="${GPU_IDS[$i]}"
    log="outputs/lsf/${LSB_JOBID:-pack}_i${idx}.log"
    echo "worker gpu=${gpu} index=${idx} log=${log}"
    INDEX="${idx}" CUDA_VISIBLE_DEVICES="${gpu}" \
        bash "${ROOT}/scripts/phase1_cell.sh" >"${log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}"
    code=$?
    if (( code > status )); then status=${code}; fi
done
echo "done=$(date) status=${status}"
exit "${status}"
