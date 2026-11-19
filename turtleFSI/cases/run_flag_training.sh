#!/usr/bin/env bash
# Run in Aero's activated turtleFSI environment. Independent serial cases.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_executable=$(command -v python)
resume_from=""
retry_options=()
if [[ ${1:-} == --resume ]]; then
    resume_from=$(realpath "${2:?Usage: --resume SOURCE [JOBS] [TIMEOUT] [FAILED_CASE_DT]}")
    test -f "$resume_from/campaign.json"
    output_parent=$(dirname "$resume_from")
    jobs=${3:-3}
    wall_limit=${4:-12h}
    if [[ -n ${5:-} ]]; then retry_options=(--retry-dt "$5"); fi
    plan=resume
else
output_parent=${1:-$HOME}
jobs=${2:-3}
end_time=${3:-30}
time_step=${4:-0.01}
wall_limit=${5:-12h}
snapshot_step=${6:-5}
plan=${7:-training16}
case "$plan" in training16|grid25) ;; *) echo 'Unknown campaign plan' >&2; exit 2 ;; esac
fi
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'jobs must be a positive integer' >&2; exit 2; }
command -v systemd-run >/dev/null
test -x /usr/bin/time
output=$(mktemp -d "$output_parent/flag-training.XXXXXX")
printf 'Results: %s\nConcurrent cases: %s\n' "$output" "$jobs"

limited() {
    systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=100% \
        bash -c '
            set -e
            group=$(awk -F: '\''$1 == "0" {print $3}'\'' /proc/self/cgroup)
            test -n "$group"
            cg="/sys/fs/cgroup$group"
            test "$(cat "$cg/memory.max")" = 2147483648
            test "$(cat "$cg/memory.swap.max")" = 0
            awk '\''$1 != "max" && $1/$2 <= 1 {ok=1} END {exit !ok}'\'' "$cg/cpu.max"
            exec "$@"
        ' limits "$@"
}

if [[ -n "$resume_from" ]]; then
    printf 'Resuming campaign cases from: %s\n' "$resume_from"
    if ! limited timeout --kill-after=30s 5m "$python_executable" \
        -m turtleFSI.cases.flag_campaign retry "$output" --resume-from "$resume_from" \
        "${retry_options[@]}" > "$output/preflight.log" 2>&1; then
        cat "$output/preflight.log"
        exit 1
    fi
    end_time=$(cat "$output/end_time.txt")
    tail -n 1 "$output/preflight.log"
elif ! limited timeout --kill-after=30s 5m "$python_executable" \
    -m turtleFSI.cases.flag_campaign prepare "$output" --end-time "$end_time" \
    --dt "$time_step" --stride "$snapshot_step" --plan "$plan" > "$output/preflight.log" 2>&1; then
    cat "$output/preflight.log"
    exit 1
fi
printf 'All maps passed. Starting %s cases (plan: %s).\n' "$(wc -l < "$output/cases.tsv")" "$plan"
pids=()
names=()
failed_any=0
printf 'case\texit_code\n' > "$output/status.tsv"
wait_batch() {
    local failed=0 i code
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then code=0; else code=$?; failed=1; fi
        printf '%s\n' "$code" > "$output/${names[$i]}.exitcode"
        printf '%s\t%s\n' "${names[$i]}" "$code" >> "$output/status.tsv"
        printf '%s: exit %s\n' "${names[$i]}" "$code"
    done
    pids=()
    names=()
    return "$failed"
}
while IFS=$'\t' read -r name radius length dt stride; do
    limited /usr/bin/time -v -o "$output/$name.resources.txt" \
        timeout --signal=TERM --kill-after=30s "$wall_limit" \
        "$python_executable" -c 'from turtleFSI.run_turtle import main; main()' \
        --problem TF_fsi_reference --end-time "$end_time" --time-step "$dt" \
        --theta 1.0 --extrapolation laplace --extrapolation-sub-type paper \
        --d-deg 2 --v-deg 2 --p-deg 1 \
        --save-step 100 --checkpoint-step 500 \
        --rom-export-dvp true --rom-export-step "$stride" \
        --folder "$output/results" --sub-folder "$name" \
        --new-arguments "R=$radius" "f_L=$length" fsi_id=5 \
        > "$output/$name.log" 2>&1 &
    pids+=("$!")
    names+=("$name")
    if (( ${#pids[@]} == jobs )); then
        wait_batch || { failed_any=1; echo 'A case failed; continuing with the next batch.'; }
    fi
done < "$output/cases.tsv"
wait_batch || failed_any=1
if (( failed_any )); then
    printf 'All queued cases attempted; some failed. See %s/status.tsv and individual logs.\n' "$output"
    exit 1
fi
if ! limited timeout --kill-after=30s 20m "$python_executable" \
    -m turtleFSI.cases.flag_campaign validate "$output" > "$output/validation.log" 2>&1; then
    cat "$output/validation.log"
    exit 1
fi
printf 'Campaign validation passed: %s/campaign_validation.json\n' "$output"
printf 'Review numerical accuracy and choose held-out data before POD training.\n'
du -sh "$output"
