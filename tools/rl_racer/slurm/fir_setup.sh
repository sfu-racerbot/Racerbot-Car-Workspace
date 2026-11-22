#!/bin/bash
# One-time install on a Fir login node with internet access.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_RACER_REPO="${RL_RACER_REPO:-$(cd "${HERE}/../../.." && pwd)}"
export RL_RACER_VENV="${RL_RACER_VENV:-${PROJECT:?set PROJECT or RL_RACER_VENV}/rl_racer/venv}"
set +u
module purge
module load StdEnv/2023 "${RL_RACER_PYTHON_MODULE:-python/3.11}" "${RL_RACER_CUDA_MODULE:-cuda/12.6}"
set -u
if [[ ! -x "${RL_RACER_VENV}/bin/python" ]]; then
    virtualenv --no-download "${RL_RACER_VENV}"
fi
source "${RL_RACER_VENV}/bin/activate"
python -m pip install -r "${RL_RACER_REPO}/tools/rl_racer/requirements.txt" pytest
# The portable requirements pin CPU torch; replace it with Fir's CUDA build.
python -m pip install --no-index --force-reinstall 'torch==2.5.1'
source "${HERE}/fir_env.sh"
python -c 'import torch, numpy, yaml; assert torch.version.cuda is not None, "Fir wheelhouse supplied CPU-only torch"; print("READY", torch.__version__, "CUDA", torch.version.cuda, "NumPy", numpy.__version__)'
cd "${RL_RACER_REPO}"
python -m pytest tools/rl_racer/tests -q
echo "Setup complete; compute jobs activate ${RL_RACER_VENV} without downloading."
