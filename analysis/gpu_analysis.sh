#!/bin/bash
# =============================================================================
# benchmark.sh — Run a CUDA denoise executable on a set of images, multiple
# times each, and record per-run timings into a CSV.
#
# USAGE:
#   ./benchmark.sh -e <executable> -n <repetitions> [-o <output.csv>]
#                  [-d <image_dir>] image1.ppm [image2.ppm ...]
#
# EXAMPLES:
#   # 10 runs of denoise_final on three images:
#   ./benchmark.sh -e ./denoise_final -n 10 noise1.ppm noise2.ppm noise3.ppm
#
#   # Same, custom output file and a directory prefix for the images:
#   ./benchmark.sh -e ./denoise_final -n 5 -o results.csv \
#                  -d /home/marranini/images   noise1.ppm noise2.ppm
#
# OUTPUT CSV COLUMNS:
#   timestamp, executable, image, repetition, fast_ms, sort_ms, slow_ms, total_ms, status
#   - timestamp  : ISO-8601 time when this run finished
#   - executable : basename of the binary that was timed
#   - image      : basename of the input image
#   - repetition : 1-based repetition index for this (executable, image) pair
#   - fast_ms    : "Pass 1 (fast)" time in ms (empty if not found)
#   - sort_ms    : "Sort" time in ms (empty if not found)
#   - slow_ms    : "Pass 2 (slow)" time in ms (empty if not found)
#   - total_ms   : "Total GPU time" in ms (empty if not found)
#   - status     : "ok" if the executable exited 0, otherwise "fail:<code>"
#
# DESIGN NOTES:
#   - The script does NOT save output images; it redirects them to /dev/null-
#     equivalent (a temp file overwritten each run) so disk I/O does not skew
#     timings on later runs.
#   - The CSV is appended to if it already exists; the header is only written
#     when the file is created fresh.  This makes it easy to combine
#     benchmark sessions for different code versions.
#   - All four timings are parsed from the program's stdout using grep+awk.
#     If a timing line is missing (e.g. an early failure) the field is empty
#     and status is set accordingly.
# =============================================================================

set -u   # treat unset variables as errors
# NOTE: we deliberately do NOT use `set -e` — we want to keep going even if
# one run fails (a single bad image shouldn't stop a 100-run benchmark).

# ---- Defaults ---------------------------------------------------------------
EXECUTABLE=""
REPETITIONS=0
OUTPUT_CSV="benchmark_results.csv"
IMAGE_DIR=""

# ---- Argument parsing -------------------------------------------------------
usage() {
    sed -n '2,30p' "$0"   # prints the header comment block as help text
    exit 1
}

while getopts ":e:n:o:d:h" opt; do
    case "$opt" in
        e) EXECUTABLE="$OPTARG" ;;
        n) REPETITIONS="$OPTARG" ;;
        o) OUTPUT_CSV="$OPTARG" ;;
        d) IMAGE_DIR="$OPTARG" ;;
        h) usage ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
        :)  echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done
shift $((OPTIND - 1))

# ---- Validation -------------------------------------------------------------
if [[ -z "$EXECUTABLE" ]]; then
    echo "ERROR: -e <executable> is required." >&2; usage
fi
if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERROR: '$EXECUTABLE' is not an executable file." >&2; exit 1
fi
if ! [[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: -n must be a positive integer (got '$REPETITIONS')." >&2; usage
fi
if [[ $# -eq 0 ]]; then
    echo "ERROR: at least one input image is required." >&2; usage
fi

IMAGES=("$@")

# Resolve image paths against IMAGE_DIR if provided
RESOLVED_IMAGES=()
for img in "${IMAGES[@]}"; do
    if [[ -n "$IMAGE_DIR" ]]; then
        path="$IMAGE_DIR/$img"
    else
        path="$img"
    fi
    if [[ ! -f "$path" ]]; then
        echo "ERROR: image not found: $path" >&2; exit 1
    fi
    RESOLVED_IMAGES+=("$path")
done

# ---- CSV setup --------------------------------------------------------------
# Write the header only if the file does not exist yet (so re-running the
# script appends to an existing benchmark log).
if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "timestamp,executable,image,repetition,fast_ms,sort_ms,slow_ms,total_ms,status" \
        > "$OUTPUT_CSV"
fi

# Temp file for each run's stdout; reused to keep disk usage minimal
TMP_LOG="$(mktemp -t denoise_run.XXXXXX.log)"
# Temp file for the output image so we don't write to a permanent location
TMP_OUT="$(mktemp -t denoise_out.XXXXXX.ppm)"
trap 'rm -f "$TMP_LOG" "$TMP_OUT"' EXIT

# ---- Helper: extract a numeric value from the program's stdout --------------
# Args:  $1 = label to grep for (e.g. "Pass 1 (fast)")
#        $2 = path to the log file
# Echoes the value in ms (just the number) or empty string if not found.
extract_ms() {
    local label="$1"
    local logfile="$2"
    # Match a line like "Pass 1 (fast) : 6.318 ms" and pull the number.
    # `awk` is more robust than `cut` for variable whitespace around the colon.
    grep -F -- "$label" "$logfile" \
        | awk -F':' '{print $2}' \
        | awk '{print $1}'
}

# ---- Main loop --------------------------------------------------------------
TOTAL_RUNS=$(( ${#RESOLVED_IMAGES[@]} * REPETITIONS ))
RUN_INDEX=0

EXEC_BASENAME="$(basename "$EXECUTABLE")"

echo "============================================================"
echo "Benchmark: $EXEC_BASENAME"
echo "Images   : ${#RESOLVED_IMAGES[@]}"
echo "Reps     : $REPETITIONS each"
echo "Total    : $TOTAL_RUNS runs"
echo "Output   : $OUTPUT_CSV"
echo "============================================================"

for img_path in "${RESOLVED_IMAGES[@]}"; do
    img_name="$(basename "$img_path")"

    for rep in $(seq 1 "$REPETITIONS"); do
        RUN_INDEX=$(( RUN_INDEX + 1 ))
        printf "[%d/%d] %s rep %d ... " \
               "$RUN_INDEX" "$TOTAL_RUNS" "$img_name" "$rep"

        # Run the program; capture stdout, ignore stderr for parsing but keep
        # the exit code for the status column.
        "$EXECUTABLE" "$img_path" "$TMP_OUT" > "$TMP_LOG" 2>&1
        exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            status="ok"
        else
            status="fail:$exit_code"
        fi

        fast_ms="$(extract_ms 'Pass 1 (fast)' "$TMP_LOG")"
        sort_ms="$(extract_ms 'Sort'          "$TMP_LOG")"
        slow_ms="$(extract_ms 'Pass 2 (slow)' "$TMP_LOG")"
        total_ms="$(extract_ms 'Total GPU time' "$TMP_LOG")"

        timestamp="$(date -Iseconds)"

        # Append one CSV row
        echo "$timestamp,$EXEC_BASENAME,$img_name,$rep,$fast_ms,$sort_ms,$slow_ms,$total_ms,$status" \
            >> "$OUTPUT_CSV"

        if [[ "$status" == "ok" ]]; then
            printf "%s ms total\n" "${total_ms:-?}"
        else
            printf "FAILED (exit %d) — see %s\n" "$exit_code" "$TMP_LOG"
            # Show the tail of the failed run so the user has a hint
            tail -n 5 "$TMP_LOG" | sed 's/^/    /'
        fi
    done
done

echo "============================================================"
echo "Done. Results appended to: $OUTPUT_CSV"
echo "Rows written this session: $TOTAL_RUNS"