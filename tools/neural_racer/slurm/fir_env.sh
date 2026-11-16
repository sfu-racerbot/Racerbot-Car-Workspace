# Sourced (not executed) by fir_train.sbatch. Loads modules and builds the
# training virtualenv on the node-local disk, the Alliance's recommended place
# for it: fast to import from, and gone when the job ends.
#
# Overrides (export before sbatch, or pass with --export=ALL,VAR=value):
#   NR_PYTHON_MODULE  python module to load          (default python/3.11)
#   NR_WHEELS         extra local wheel directory for packages the
#                     wheelhouse lacks (built once on a login node, README)
#   NR_SKIP_ENV=1     use whatever python is already active -- local dry runs
#                     only, never on FIR

if [[ "${NR_SKIP_ENV:-0}" == "1" ]]; then
    echo "[nr-env] NR_SKIP_ENV=1: using $(command -v python3)"
    return 0
fi

: "${SLURM_TMPDIR:?fir_env.sh must run inside a Slurm job}"

# Lmod's scripts reference unset variables; relax nounset around them.
set +u
module purge
module load StdEnv/2023 cuda "${NR_PYTHON_MODULE:-python/3.11}"
set -u

NR_VENV="${SLURM_TMPDIR}/nr-venv"
virtualenv --no-download "${NR_VENV}"
# shellcheck disable=SC1091
source "${NR_VENV}/bin/activate"
pip install --no-index --upgrade pip

pip_args=(--no-index -r "${NR_REPO}/tools/neural_racer/requirements-fir.txt")
if [[ -n "${NR_WHEELS:-}" ]]; then
    pip_args+=(--find-links "${NR_WHEELS}")
fi
pip install "${pip_args[@]}"

if ! python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'; \
print('[nr-env] torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"; then
    mkdir -p "${SCRATCH}/neural_racer"
    echo "$(date +%s) ${SLURMD_NODENAME:-$(hostname -s)}" >> "${SCRATCH}/neural_racer/bad_nodes.log"
    echo "[nr-env] Quarantined broken GPU node ${SLURMD_NODENAME:-$(hostname -s)} for 24h"
    exit 1
fi
# The LiDAR cast's backend: the fallback is silent, and it cost every run until 2026-10-29 ~6x
python -c "import triton; print('[nr-env] triton', triton.__version__, '(the LiDAR cast)')" 2>/dev/null \
    || echo "[nr-env] WARNING: no triton; the LiDAR cast falls back to PyTorch (~6x slower)"
