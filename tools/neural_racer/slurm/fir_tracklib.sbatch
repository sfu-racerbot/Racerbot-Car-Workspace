#!/bin/bash
# Build (or top up) the training track library on CPU nodes, before any
# training run: a run then fills its whole GPU pool at start-up.
#
#   sbatch --account=def-<pi> tools/neural_racer/slurm/fir_tracklib.sbatch [DIR] [COUNT] [CONFIG] [REAL_MAPS]
#
# DIR defaults to $SCRATCH/neural_racer/tracklib-v1, COUNT to 8192 (two
# pools of 4096), CONFIG to v1 (learn.TRAIN_CONFIGS; v2 adds the tight
# corners the real maps have; v3 hairpin loops), REAL_MAPS (comma-separated) adds every variant of
# those real maps (tracklib --real). Resumable: rerun to finish an interrupted build.
#SBATCH --job-name=nr-tracklib
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

OUT="${1:-${SCRATCH:?}/neural_racer/tracklib-v1}"
COUNT="${2:-8192}"
CONFIG="${3:-v1}"
REAL="${4:-}"
NR_REPO="${NR_REPO:-${SLURM_SUBMIT_DIR:?submit from the workspace root}}"

set +u; module purge; module load StdEnv/2023 "${NR_PYTHON_MODULE:-python/3.11}"; set -u
virtualenv --no-download -q "${SLURM_TMPDIR}/venv" && source "${SLURM_TMPDIR}/venv/bin/activate"
pip install -q --no-index -r "${NR_REPO}/tools/neural_racer/requirements-fir.txt"
export PYTHONPATH="${NR_REPO}/tools/neural_racer:${NR_REPO}/src/neural_racer:${NR_REPO}/src/gap_follow:${PYTHONPATH:-}"
cd "${NR_REPO}"
OMP_NUM_THREADS=1 python -u -m neural_racer_train.tracklib --out "${OUT}" --count "${COUNT}" --config "${CONFIG}" --real "${REAL}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
