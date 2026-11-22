#!/bin/bash
# Submit from the workspace root; no job is submitted by this file itself.
# sbatch --account=def-<pi> tools/rl_racer/slurm/train_paper.sbatch /path/to/signoff.yaml run-name
# One Fir H100's share: 12 CPU cores, 64 GiB; NumPy simulation remains CPU-bound.
# 12 hours is an unmeasured segment estimate, not a 120M-step completion promise.
#SBATCH --job-name=rl-paper
#SBATCH --gpus-per-node=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --requeue

set -euo pipefail
SIGNOFF="${1:?reviewer signoff YAML is mandatory}"
RUN_NAME="${2:?run name is mandatory}"
[[ "${RUN_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo 'Invalid run name' >&2; exit 2; }
SIGNOFF="$(realpath "${SIGNOFF}")"
export RL_RACER_REPO="${RL_RACER_REPO:-${SLURM_SUBMIT_DIR:?submit from workspace root}}"
CONFIG="${RL_RACER_CONFIG:-${RL_RACER_REPO}/tools/rl_racer/config/train_paper.yaml}"
export RL_RACER_REQUIRE_CUDA=1
source "${RL_RACER_REPO}/tools/rl_racer/slurm/fir_env.sh"
python -c 'import sys; from rl_racer.train import load_config, validate_signoff; c = load_config(sys.argv[1]); validate_signoff(sys.argv[1], c["total_steps"], sys.argv[2])' "${CONFIG}" "${SIGNOFF}"
RUN_DIR="${RL_RACER_RUNS_ROOT:-${SCRATCH:?set SCRATCH or RL_RACER_RUNS_ROOT}/rl_racer/runs}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/slurm-${SLURM_JOB_ID}.log") 2>&1
echo "Run: ${RUN_DIR}; resuming the last completed checkpoint if present"
ARGS=(--config "${CONFIG}" --signoff "${SIGNOFF}" --run-dir "${RUN_DIR}")
if [[ -f "${RUN_DIR}/latest.pt" ]]; then
    ARGS+=(--resume "${RUN_DIR}/latest.pt")
fi
exec python -u -m rl_racer.train "${ARGS[@]}"
