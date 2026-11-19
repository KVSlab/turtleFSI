#!/usr/bin/env bash
# Run from the activated server Conda environment. No installation is changed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_executable=$(command -v python)
command -v systemd-run >/dev/null
test -x /usr/bin/time
# Pass a parent directory to keep results outside /tmp, e.g. /home/ml49985.
output_parent=${1:-${TMPDIR:-/tmp}}
problem=${2:-TF_fsi_geometry}
end_time=${3:-0.1}
time_step=${4:-0.01}
wall_limit=${5:-10m}
snapshot_step=${6:-1}
case "$problem" in
    TF_fsi_geometry|TF_fsi_reference) ;;
    *) printf 'Unknown flag problem: %s\n' "$problem" >&2; exit 2 ;;
esac
output=$(mktemp -d "$output_parent/flag-geometry-tests.XXXXXX")
printf 'Results: %s\n' "$output"
printf 'Problem: %s\n' "$problem"
printf 'End time: %s; dt: %s; timeout: %s; snapshot stride: %s\n' "$end_time" "$time_step" "$wall_limit" "$snapshot_step"

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
            echo "Verified limits: 2 GiB RAM, zero swap, at most one CPU."
            exec "$@"
        ' limits "$@"
}

# Mapping validation finishes before any concurrent simulation starts.
if ! limited timeout --kill-after=30s 5m "$python_executable" \
    -m turtleFSI.cases.validate_flag_geometry --output-dir "$output/meshes" \
    > "$output/validation.log" 2>&1; then
    cat "$output/validation.log"
    exit 1
fi
printf 'Geometry validation passed. Starting three cases.\n'

names=(baseline longer_flag larger_circle)
radii=(0.05 0.05 0.0525)
lengths=(0.35 0.3675 0.35)
pids=()
for i in 0 1 2; do
    name=${names[$i]}
    limited /usr/bin/time -v -o "$output/$name.resources.txt" \
        timeout --signal=TERM --kill-after=30s "$wall_limit" \
        "$python_executable" -c 'from turtleFSI.run_turtle import main; main()' \
        --problem "$problem" --end-time "$end_time" --time-step "$time_step" \
        --theta 1.0 --extrapolation laplace --extrapolation-sub-type paper \
        --d-deg 2 --v-deg 2 --p-deg 1 \
        --save-step 10 --checkpoint-step 10000 \
        --rom-export-dvp true --rom-export-step "$snapshot_step" \
        --folder "$output/results" --sub-folder "$name" \
        --new-arguments "R=${radii[$i]}" "f_L=${lengths[$i]}" fsi_id=5 \
        > "$output/$name.log" 2>&1 &
    pids+=("$!")
done

failed=0
for i in 0 1 2; do
    if wait "${pids[$i]}"; then
        printf '%s: finished\n' "${names[$i]}"
    else
        printf '%s: FAILED (see %s/%s.log)\n' "${names[$i]}" "$output" "${names[$i]}"
        failed=1
    fi
done
for name in "${names[@]}"; do
    if [[ -f "$output/$name.resources.txt" ]]; then
        printf '\n%s resources:\n' "$name"
        grep -E 'Elapsed|Maximum resident|Exit status' "$output/$name.resources.txt" || true
    fi
done
du -sh "$output"
exit "$failed"
