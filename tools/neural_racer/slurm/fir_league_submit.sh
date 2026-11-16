#!/bin/bash
# Submit the Stage 3 league to FIR: one 1-GPU learner per role, each a chain of
# resumable segments. Run from the workspace root on a FIR login node.
#
#   tools/neural_racer/slurm/fir_league_submit.sh --account def-<pi> --run <name> \
#       --init <student checkpoint> [--init-mlp <MLP student checkpoint>] \
#       [--teacher <teacher best.pt>] [--critic-from <teacher run dir>] [--library <tracklib>] \
#       [--roles "main0 main1_mlp ..."] [--segments 8] [--time 03:00:00] [--num-envs N] [--exclude=NODES] \
#       [--dry-run] \
#       [-- <more league_learner args>]
#
# Every role shares <run>/archive, so snapshots from any role become opponents
# for all of them. Segments are chained with afterany: a segment that fails or
# is pre-empted still lets the next one start and resume from the checkpoint.
# A role that has finished writes DONE and its later segments exit in seconds.
# main1_mlp starts from --init-mlp (distillation's --memory none student): the
# GRU student run without its memory is not the ablation. 3-hour segments start
# far sooner on FIR than 12-hour ones.
set -euo pipefail

ROLES="main0 main1_mlp main_exploiter league_exploiter"
ACCOUNT="" RUN="" INIT="" INIT_MLP="" SEGMENTS=8 TIME="03:00:00" DRY_RUN=0 NUM_ENVS=""
EXTRA=() SB=()

while (( $# )); do
    case "$1" in
        --account)     ACCOUNT="$2"; shift 2 ;;
        --run)         RUN="$2"; shift 2 ;;
        --init)        INIT="$2"; shift 2 ;;
        --init-mlp)    INIT_MLP="$2"; shift 2 ;;
        --teacher)     EXTRA+=(--teacher "$2"); shift 2 ;;
        --critic-from) EXTRA+=(--critic-from "$2"); shift 2 ;;
        --library)     EXTRA+=(--library "$2"); shift 2 ;;
        --roles)       ROLES="$2"; shift 2 ;;
        --segments)    SEGMENTS="$2"; shift 2 ;;
        --time)        TIME="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --exclude=*)   SB+=("$1"); shift ;;
        --num-envs)    NUM_ENVS="$2"; shift 2 ;;
        --)            shift; EXTRA+=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "${ACCOUNT}" ]] || { echo "--account def-<pi> is required" >&2; exit 2; }
[[ -n "${RUN}" ]]     || { echo "--run <name> is required" >&2; exit 2; }
[[ -n "${INIT}" ]]    || { echo "--init <Stage 2 student checkpoint> is required" >&2; exit 2; }
[[ "${SEGMENTS}" =~ ^[1-9][0-9]*$ ]] || { echo "--segments must be a positive integer" >&2; exit 2; }
if [[ " ${ROLES} " == *" main1_mlp "* && -z "${INIT_MLP}" ]]; then
    echo "main1_mlp needs --init-mlp (distillation's --memory none student), or leave it out of --roles" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRATCH:?}/neural_racer/logs"        # job logs beside the runs, not in the workspace
mkdir -p "${LOG_DIR}"
REPO="$(cd "${HERE}/../../.." && pwd)"

for role in ${ROLES}; do
    init="${INIT}"; [[ "${role}" == *mlp* ]] && init="${INIT_MLP}"
    prev=""
    for (( segment = 1; segment <= SEGMENTS; segment++ )); do
        cmd=(sbatch --parsable --account "${ACCOUNT}" --time "${TIME}" --output "${LOG_DIR}/%x-%j.out"
             --job-name "nr-${RUN}-${role}"
             --export "ALL,NR_ROLE=${role},NR_REPO=${REPO}" "${SB[@]}")
        [[ -n "${prev}" ]] && cmd+=(--dependency "afterany:${prev}")
        cmd+=("${HERE}/fir_train.sbatch" league_learner "${RUN}" --init-from "${init}")
        [[ -n "${NUM_ENVS}" ]] && cmd+=(--num-envs "${NUM_ENVS}")
        cmd+=("${EXTRA[@]}")
        if (( DRY_RUN )); then
            echo "${cmd[*]}"
            prev="DRYRUN_${role}_${segment}"
        else
            prev="$("${cmd[@]}")"
            echo "${role} segment ${segment}/${SEGMENTS}: job ${prev}"
        fi
    done
done
