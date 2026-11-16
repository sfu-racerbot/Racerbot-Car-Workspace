#!/bin/bash
set -e

ACCOUNT="${SBATCH_ACCOUNT:-def-<yourprof>}"
RUN_DIR="$SCRATCH/neural_racer/runs"
BENCH_JSON="$RUN_DIR/bench-v1/solo/benchmark.json"
BAD_NODES_LOG="$SCRATCH/neural_racer/bad_nodes.log"
TRACKLIB="$SCRATCH/neural_racer/tracklib-v1"
TEACHER_RUN=teacher-v3
TEACHER_ENVS=32768      # no LiDAR in the teacher's sim: 672k steps/s at this count on an H100

cd ~/racerbot-ws

# Job logs go to scratch beside the runs, never into the workspace or home
LOG_DIR="$SCRATCH/neural_racer/logs"
mkdir -p "$LOG_DIR"
sbatch() { command sbatch --output="$LOG_DIR/%x-%j.out" "$@"; }

get_envs() {
    if [ ! -f "$BENCH_JSON" ]; then
        echo "Error: Benchmark not found at $BENCH_JSON. Run './run.sh bench' first."
        exit 1
    fi
    python3 -c "import json; print(json.load(open('$BENCH_JSON'))['chosen']['num_envs'])"
}

EXCL_ARGS=()
if [ -f "$BAD_NODES_LOG" ]; then
    # 7 days: fc10512 (no CUDA device) was quarantined on 2026-10-29, back after 24 h, and failed six jobs on 10-31
    BAD_LIST=$(awk -v now="$(date +%s)" '(now - $1) < 604800 && !seen[$2]++ { out = (out ? out "," : "") $2 } END { print out }' "$BAD_NODES_LOG")
    if [ -n "$BAD_LIST" ]; then
        echo "==> Skipping broken GPU nodes (7-day auto-expiry): $BAD_LIST"
        EXCL_ARGS=(--exclude="$BAD_LIST")
    fi
fi

case "$1" in
    bench)
        echo "==> Submitting benchmark (30 min limit)..."
        sbatch --account="$ACCOUNT" --time=00:30:00 "${EXCL_ARGS[@]}" \
            tools/neural_racer/slurm/fir_train.sbatch benchmark bench-v1
        ;;

    envs)
        echo "Chosen num_envs: $(get_envs)"
        ;;

    tracklib)
        # ./run.sh tracklib [v1|v2|v3] [REAL_MAPS]: v2 (learn.TRAIN_CONFIGS) adds the tight corners every
        # real map has, v3 hairpin loops and, by default, car 1's course itself (every variant, 40 copies)
        CFG="${2:-v1}"; OUT="$SCRATCH/neural_racer/tracklib-$CFG"
        REAL="${3:-}"; [ "$CFG" = v3 ] && [ -z "$REAL" ] && REAL=car1_course_2026_10_26
        echo "==> Submitting the $CFG track library build (CPU) into $OUT${REAL:+, with $REAL}..."
        sbatch --account="$ACCOUNT" tools/neural_racer/slurm/fir_tracklib.sbatch "$OUT" 8192 "$CFG" "$REAL"
        ;;
    teacher)
        # ./run.sh teacher [seeds...] [after=JOBID]: default seeds 0 1 2; after= waits for a library build
        SEEDS=(); AFTER=""
        for a in "${@:2}"; do
            case "$a" in after=*) AFTER="${a#after=}" ;; *) SEEDS+=("$a") ;; esac
        done
        [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0 1 2)
        if [ -z "$AFTER" ] && [ ! -f "$TRACKLIB/library.json" ]; then
            echo "Error: no track library at $TRACKLIB. Run './run.sh tracklib' first (or pass after=<its job id>)."
            exit 1
        fi
        echo "==> Submitting teacher seeds ${SEEDS[*]} (2 chained segments each), num_envs=$TEACHER_ENVS..."
        for s in "${SEEDS[@]}"; do
            DEP=(); [ -n "$AFTER" ] && DEP=(--dependency=afterok:$AFTER)
            prev=$(sbatch --parsable --account="$ACCOUNT" "${EXCL_ARGS[@]}" "${DEP[@]}" --export=ALL,NR_ROLE=seed$s \
                tools/neural_racer/slurm/fir_train.sbatch teacher $TEACHER_RUN --seed $s --num-envs $TEACHER_ENVS \
                --library "$TRACKLIB")
            echo "seed $s: job $prev"
            prev=$(sbatch --parsable --dependency=afterok:$prev --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch teacher $TEACHER_RUN --seed $s \
                --num-envs $TEACHER_ENVS --library "$TRACKLIB")
            echo "seed $s: job $prev (resumes if the first segment runs out of time)"
        done
        echo "All seeds queued successfully."
        ;;

    teacher4)
        # ./run.sh teacher4 [squash|penalty|masked] [seeds...]: teacher v4, fine-tuned from teacher-v3 seed 0
        # inside the car's cornering cap (training plan step 1); penalty is item 2.1's A/B, masked
        # hides the unseen track in half the episodes' reactive laps (item 2.3, decision D12)
        MODE="${2:-squash}"; SEEDS=("${@:3}"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0 1 2)
        NAME=teacher-v4; [ "$MODE" = squash ] || NAME=teacher-v4-$MODE
        EXTRA=(); [ "$MODE" = masked ] && { MODE=squash; EXTRA=(--mask-share 0.5); }
        INIT="$RUN_DIR/$TEACHER_RUN/seed0"
        [ -f "$INIT/checkpoints/best.pt" ] || { echo "Error: no $INIT/checkpoints/best.pt"; exit 1; }
        echo "==> Submitting $NAME ($MODE) seeds ${SEEDS[*]} from $INIT, num_envs=$TEACHER_ENVS..."
        for s in "${SEEDS[@]}"; do
            ARGS=(teacher $NAME --seed $s --num-envs $TEACHER_ENVS --library "$TRACKLIB" --init-from "$INIT"
                  --steps 1e9 --cap-mode "$MODE" "${EXTRA[@]}")
            prev=$(sbatch --parsable --time=03:00:00 --account="$ACCOUNT" "${EXCL_ARGS[@]}" --export=ALL,NR_ROLE=seed$s \
                tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev"
            prev=$(sbatch --parsable --time=03:00:00 --dependency=afterok:$prev --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev (resumes if the first segment runs out of time)"
        done
        ;;
    teacher5)
        # ./run.sh teacher5 [plain|masked] [seeds...]: teacher v5, fine-tuned from teacher v4 inside the car's
        # whole safety wrapper (--safety on: the LiDAR braking cap and its side margin, which v4 never met --
        # in the student's races it stalled against the margin or braked into a wall in 7 of 8). plain starts
        # from v4 seed 1 (the sweep's teacher), masked from teacher-v4-masked. Its races cast the LiDAR, so
        # its held-out races run on the GPU between updates
        MODE="${2:-plain}"; SEEDS=("${@:3}"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0)
        NAME=teacher-v5; INIT="$RUN_DIR/teacher-v4/seed1"; EXTRA=()
        [ "$MODE" = masked ] && { NAME=teacher-v5-masked; INIT="$RUN_DIR/teacher-v4-masked/seed0"; EXTRA=(--mask-share 0.5); }
        [ -f "$INIT/checkpoints/best.pt" ] || { echo "Error: no $INIT/checkpoints/best.pt"; exit 1; }
        echo "==> Submitting $NAME seeds ${SEEDS[*]} from $INIT, num_envs=$TEACHER_ENVS..."
        for s in "${SEEDS[@]}"; do
            ARGS=(teacher $NAME --seed $s --num-envs $TEACHER_ENVS --library "$TRACKLIB" --init-from "$INIT"
                  --steps 5e8 --cap-mode squash --safety on "${EXTRA[@]}")
            prev=$(sbatch --parsable --time=03:00:00 --account="$ACCOUNT" "${EXCL_ARGS[@]}" --export=ALL,NR_ROLE=seed$s \
                tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev"
            prev=$(sbatch --parsable --time=03:00:00 --dependency=afterok:$prev --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev (resumes if the first segment runs out of time)"
        done
        ;;
    teacher6)
        # ./run.sh teacher6 [plain|masked] [seeds...]: teacher v6, fine-tuned from teacher v5 on the v2
        # track library (corners down to 0.45 m that the car must swing wide for). v5 and its students
        # stalled in every race on car 1's course, whose corners v1's library never had. Its held-out
        # races include the obstacle-free real maps (--eval-real), car 1's course among them
        MODE="${2:-plain}"; SEEDS=("${@:3}"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0)
        LIB2="$SCRATCH/neural_racer/tracklib-v2"
        [ -f "$LIB2/library.json" ] || { echo "Error: no $LIB2. Run './run.sh tracklib v2' first."; exit 1; }
        NAME=teacher-v6; INIT="$RUN_DIR/teacher-v5/seed0"; EXTRA=()
        [ "$MODE" = masked ] && { NAME=teacher-v6-masked; INIT="$RUN_DIR/teacher-v5-masked/seed0"; EXTRA=(--mask-share 0.5); }
        [ -f "$INIT/checkpoints/best.pt" ] || { echo "Error: no $INIT/checkpoints/best.pt"; exit 1; }
        echo "==> Submitting $NAME seeds ${SEEDS[*]} from $INIT on $LIB2, num_envs=$TEACHER_ENVS..."
        for s in "${SEEDS[@]}"; do
            # a wrapper-on eval takes 13-29 min on the GPU (2026-11-01): every 100 updates, and the
            # time-limit signal 25 min early so the races under way can finish
            ARGS=(teacher $NAME --seed $s --num-envs $TEACHER_ENVS --library "$LIB2" --init-from "$INIT"
                  --steps 5e8 --cap-mode squash --safety on --eval-real --eval-every 100 "${EXTRA[@]}")
            prev=$(sbatch --parsable --time=03:00:00 --signal=B:USR1@1500 --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev"
            prev=$(sbatch --parsable --time=03:00:00 --signal=B:USR1@1500 --dependency=afterok:$prev \
                --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev (resumes if the first segment runs out of time)"
        done
        ;;
    teacher7)
        # ./run.sh teacher7 [plain|masked] [seeds...]: teacher v7, fine-tuned from teacher v6 on the v3
        # library: hairpin loops, and car 1's course itself (mirrored, reversed, scaled). v6 and its
        # students stalled at car 1's hairpin exit in every race there (2026-11-01)
        MODE="${2:-plain}"; SEEDS=("${@:3}"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0)
        LIB3="$SCRATCH/neural_racer/tracklib-v3"
        [ -f "$LIB3/library.json" ] || { echo "Error: no $LIB3. Run './run.sh tracklib v3' first."; exit 1; }
        NAME=teacher-v7; INIT="$RUN_DIR/teacher-v6/seed0"; EXTRA=()
        [ "$MODE" = masked ] && { NAME=teacher-v7-masked; INIT="$RUN_DIR/teacher-v6-masked/seed0"; EXTRA=(--mask-share 0.5); }
        [ -f "$INIT/checkpoints/best.pt" ] || { echo "Error: no $INIT/checkpoints/best.pt"; exit 1; }
        echo "==> Submitting $NAME seeds ${SEEDS[*]} from $INIT on $LIB3, num_envs=$TEACHER_ENVS..."
        for s in "${SEEDS[@]}"; do
            ARGS=(teacher $NAME --seed $s --num-envs $TEACHER_ENVS --library "$LIB3" --init-from "$INIT"
                  --steps 5e8 --cap-mode squash --safety on --eval-real --eval-every 100 "${EXTRA[@]}")
            prev=$(sbatch --parsable --time=03:00:00 --signal=B:USR1@1500 --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev"
            prev=$(sbatch --parsable --time=03:00:00 --signal=B:USR1@1500 --dependency=afterok:$prev \
                --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev (resumes if the first segment runs out of time)"
        done
        ;;
    teacher8)
        # ./run.sh teacher8 [plain|masked] [seeds...]: teacher v8, teacher v7 trained on, on the v3 library,
        # from v7's last actor (--init-actor latest): v7's best.pt is its update 100, but car 1's course
        # kept improving to its last update (rerace finish 0.22 -> 0.30, 2026-11-02)
        MODE="${2:-plain}"; SEEDS=("${@:3}"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0)
        LIB3="$SCRATCH/neural_racer/tracklib-v3"
        [ -f "$LIB3/library.json" ] || { echo "Error: no $LIB3. Run './run.sh tracklib v3' first."; exit 1; }
        NAME=teacher-v8; INIT="$RUN_DIR/teacher-v7/seed0"; EXTRA=()
        [ "$MODE" = masked ] && { NAME=teacher-v8-masked; INIT="$RUN_DIR/teacher-v7-masked/seed0"; EXTRA=(--mask-share 0.5); }
        [ -f "$INIT/checkpoints/best.pt" ] || { echo "Error: no $INIT/checkpoints/best.pt"; exit 1; }
        echo "==> Submitting $NAME seeds ${SEEDS[*]} from $INIT on $LIB3, num_envs=$TEACHER_ENVS..."
        for s in "${SEEDS[@]}"; do
            ARGS=(teacher $NAME --seed $s --num-envs $TEACHER_ENVS --library "$LIB3" --init-from "$INIT" --init-actor latest
                  --steps 5e8 --cap-mode squash --safety on --eval-real --eval-every 100 "${EXTRA[@]}")
            prev=$(sbatch --parsable --time=03:00:00 --signal=B:USR1@1500 --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev"
            prev=$(sbatch --parsable --time=03:00:00 --signal=B:USR1@1500 --dependency=afterok:$prev \
                --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev (resumes if the first segment runs out of time)"
        done
        ;;
    teacher9)
        # ./run.sh teacher9 [plain|masked] [seeds...]: teacher v9, from teacher v8's best.pt, on tracklib-v3r10
        # (v3 with 10 copies of each car 1 variant, not 40: v8 memorised car 1's course, 0.36 -> 0.64, while
        # car 2's fell 0.56 -> 0.24), with the floor's grip as measured (mu 0.30-0.80) and the car's real
        # top speed (5.05 m/s since car 1's motor gain fix), 2026-11-03
        MODE="${2:-plain}"; SEEDS=("${@:3}"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0)
        LIB3="$SCRATCH/neural_racer/tracklib-v3r10"
        [ -f "$LIB3/library.json" ] || { echo "Error: no $LIB3. Build it: python -m neural_racer_train.tracklib --out $LIB3 --count 8192 --config v3 --real car1_course_2026_10_26 --real-copies 10"; exit 1; }
        NAME=teacher-v9; INIT="$RUN_DIR/teacher-v8/seed0"; EXTRA=()
        [ "$MODE" = masked ] && { NAME=teacher-v9-masked; INIT="$RUN_DIR/teacher-v8-masked/seed0"; EXTRA=(--mask-share 0.5); }
        [ -f "$INIT/checkpoints/best.pt" ] || { echo "Error: no $INIT/checkpoints/best.pt"; exit 1; }
        echo "==> Submitting $NAME seeds ${SEEDS[*]} from $INIT on $LIB3, num_envs=$TEACHER_ENVS..."
        for s in "${SEEDS[@]}"; do
            ARGS=(teacher $NAME --seed $s --num-envs $TEACHER_ENVS --library "$LIB3" --init-from "$INIT"
                  --steps 5e8 --cap-mode squash --safety on --eval-real --eval-every 100 "${EXTRA[@]}")
            prev=$(sbatch --parsable --time=03:00:00 --signal=B:USR1@1500 --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev"
            prev=$(sbatch --parsable --time=03:00:00 --signal=B:USR1@1500 --dependency=afterok:$prev \
                --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=seed$s tools/neural_racer/slurm/fir_train.sbatch "${ARGS[@]}")
            echo "seed $s: job $prev (resumes if the first segment runs out of time)"
        done
        ;;
    distill-sweep)
        # ./run.sh distill-sweep TEACHER_CKPT [MASKED_TEACHER_CKPT] [STEPS]: the provisional (pre-G0,
        # decision D8) Stage 2 sweep of plan items 2.3/2.5/2.6, solo races so the solo teacher's
        # labels hold. v3: the wrapper's braking cap on the swept path (the v2 runs trained where it
        # held every car to ~0.3 m/s). No traffic run: the solo teacher cannot race in traffic (it
        # stalled behind opponents in 40% of races, 2026-10-30 audit), so that waits for a teacher
        # trained in traffic. full and masked are submitted first: they start first in the queue.
        # [LIB] [TAG] [VARIANTS]: the track library (default v1's), the run-name tag (distill-<TAG>-*,
        # default v3) and which runs ("full half wide mlp latent masked", default all). MASKED "-" = none.
        # The teacher's reference bar is 0.7: it cannot see two real maps' boxes, so it tops out near 0.8
        T="${2:?teacher checkpoint}"; TM="${3:-}"; [ "$TM" = - ] && TM=""; STEPS="${4:-2e8}"
        LIB="${5:-$TRACKLIB}"; TAG="${6:-v3}"; VARIANTS=" ${7:-full half wide mlp latent masked} "
        [ -f "$T" ] || { echo "Error: no $T"; exit 1; }
        [ -f "$LIB/library.json" ] || { echo "Error: no track library at $LIB"; exit 1; }
        ENVS=8192
        COMMON=(--teacher "$T" --num-envs $ENVS --library "$LIB" --steps "$STEPS" --min-teacher-finish 0.7)
        want() { [[ "$VARIANTS" == *" $1 "* ]]; }
        submit() {   # submit NAME [distill args...]: three chained 3-hour segments (a 3 h limit backfills sooner)
            local name=$1; shift
            local j=$(sbatch --parsable --time=03:00:00 --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                --export=ALL,NR_ROLE=solo tools/neural_racer/slurm/fir_train.sbatch distill "$name" "$@")
            echo "$name: job $j"
            for _ in 1 2; do
                j=$(sbatch --parsable --time=03:00:00 --dependency=afterok:$j --account="$ACCOUNT" "${EXCL_ARGS[@]}" \
                    --export=ALL,NR_ROLE=solo tools/neural_racer/slurm/fir_train.sbatch distill "$name" "$@")
            done
        }
        want full && submit distill-$TAG-full "${COMMON[@]}" --max-cars 1
        if [ -n "$TM" ] && want masked; then
            submit distill-$TAG-masked --teacher "$TM" --num-envs $ENVS --library "$LIB" --steps "$STEPS" \
                --min-teacher-finish 0.7 --max-cars 1 --teacher-mask
        fi
        want half && submit distill-$TAG-half "${COMMON[@]}" --max-cars 1 --width half
        want wide && submit distill-$TAG-wide "${COMMON[@]}" --max-cars 1 --width wide
        want mlp && submit distill-$TAG-mlp "${COMMON[@]}" --max-cars 1 --memory none
        want latent && submit distill-$TAG-latent "${COMMON[@]}" --max-cars 1 --latent 0.1
        true
        ;;
    distill)
        SEED="${2:-0}"
        ENVS=$(get_envs)
        TEACHER_CKPT="$RUN_DIR/$TEACHER_RUN/seed${SEED}/checkpoints/best.pt"     # best held-out race
        if [ ! -f "$TEACHER_CKPT" ]; then
            echo "Error: Checkpoint $TEACHER_CKPT does not exist yet."
            exit 1
        fi
        echo "==> Submitting Distillation with Teacher Seed $SEED (num_envs=$ENVS)..."
        sbatch --account="$ACCOUNT" "${EXCL_ARGS[@]}" tools/neural_racer/slurm/fir_train.sbatch distill distill-v1 \
            --teacher "$TEACHER_CKPT" --num-envs $ENVS
        ;;

    league)
        # ./run.sh league STUDENT MLP_STUDENT TEACHER [RUN] [ROLES] [SEGMENTS] [LIB]: Stage 3 from the distilled
        # students (best.pt of distill-v3-full and of distill-v3-mlp, for main1_mlp; - with ROLES that
        # leave main1_mlp out) and the teacher they learned from: its labels kickstart, and its run's
        # critic starts the learners' critic. ROLES main0 SEGMENTS 1 is the 3-hour GPU smoke. LIB: the
        # track library, default tracklib-v3 (v1 has none of the tight corners, hairpins or car 1's course)
        S="${2:?student checkpoint}"; SM="${3:?MLP student checkpoint, or -}"; T="${4:?teacher checkpoint}"
        RUN="${5:-league-v1}"; ROLES="${6:-main0 main1_mlp main_exploiter league_exploiter}"; SEG="${7:-8}"
        LLIB="${8:-$SCRATCH/neural_racer/tracklib-v3}"
        for f in "$S" "$T"; do [ -f "$f" ] || { echo "Error: no $f"; exit 1; }; done
        [ -f "$LLIB/library.json" ] || { echo "Error: no track library at $LLIB"; exit 1; }
        MLP=(); [ "$SM" != - ] && MLP=(--init-mlp "$SM")
        echo "==> Submitting the league $RUN: roles $ROLES, $SEG segments of 3 h each..."
        tools/neural_racer/slurm/fir_league_submit.sh --account "$ACCOUNT" --run "$RUN" --init "$S" "${MLP[@]}" \
            --teacher "$T" --critic-from "$(dirname "$(dirname "$T")")" --library "$LLIB" \
            --roles "$ROLES" --segments "$SEG" --time 03:00:00 "${EXCL_ARGS[@]}"
        ;;

    status)
        echo "--- Active Slurm Jobs ---"
        squeue -u $USER
        echo ""
        echo "--- Failure Check ---"
        find "$RUN_DIR" -name "FAILED" 2>/dev/null || echo "No FAILED markers found."
        ;;

    *)
        echo "Usage: ./run.sh {bench|envs|teacher [seeds] [after=JOBID]|teacher4 [squash|penalty|masked] [seeds]|teacher5 [plain|masked] [seeds]|teacher6 [plain|masked] [seeds]|teacher7 [plain|masked] [seeds]|teacher8 [plain|masked] [seeds]|teacher9 [plain|masked] [seeds]|tracklib [v1|v2|v3] [REAL_MAPS]|distill-sweep TEACHER [MASKED|-] [STEPS] [LIB] [TAG] [VARIANTS]|distill [seed_num]|league STUDENT MLP_STUDENT TEACHER [RUN] [ROLES] [SEGMENTS] [LIB]|status}"
        exit 1
        ;;
esac
