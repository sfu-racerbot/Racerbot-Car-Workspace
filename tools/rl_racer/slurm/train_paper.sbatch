#!/bin/bash
# Submit from the workspace root; no job is submitted by this file itself.
# sbatch --account=def-<pi> tools/rl_racer/slurm/train_paper.sbatch /path/to/signoff.yaml /scratch/run-dir
#SBATCH --job-name=rl-paper
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
SIGNOFF="${1:?reviewer signoff YAML is mandatory}"
RUN_DIR="${2:?run output directory is mandatory}"
REPO="${RL_RACER_REPO:-${SLURM_SUBMIT_DIR:?submit from workspace root}}"
PYTHON="${RL_RACER_PYTHON:-${REPO}/.venv-f1tenth/bin/python}"
CONFIG="${RL_RACER_CONFIG:-${REPO}/tools/rl_racer/config/train_paper.yaml}"
export PYTHONPATH="${REPO}/tools/rl_racer${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
ARGS=(--config "${CONFIG}" --signoff "${SIGNOFF}" --run-dir "${RUN_DIR}")
if [[ -f "${RUN_DIR}/latest.pt" ]]; then
    ARGS+=(--resume "${RUN_DIR}/latest.pt")
fi
exec "${PYTHON}" -m rl_racer.train "${ARGS[@]}"
