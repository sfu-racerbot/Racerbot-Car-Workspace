# Source from Fir jobs or a login shell; never installs on compute nodes.
set +u
module purge
module load StdEnv/2023 "${RL_RACER_PYTHON_MODULE:-python/3.11}" "${RL_RACER_CUDA_MODULE:-cuda/12.6}"
set -u

RL_RACER_VENV="${RL_RACER_VENV:-${PROJECT:?set PROJECT or RL_RACER_VENV}/rl_racer/venv}"
if [[ ! -x "${RL_RACER_VENV}/bin/python" ]]; then
    echo "Run bash tools/rl_racer/slurm/fir_setup.sh on a login node first" >&2
    return 1
fi
source "${RL_RACER_VENV}/bin/activate"
export PYTHONPATH="${RL_RACER_REPO:?set RL_RACER_REPO}/tools/rl_racer${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python -c 'import os, torch; print("torch", torch.__version__); assert not os.environ.get("RL_RACER_REQUIRE_CUDA") or torch.cuda.is_available(), "CUDA unavailable: refuse GPU allocation with CPU-only training"'
