#!/usr/bin/env bash
# Reproduce every row of the evaluation tables in the README.
#
#   eval/run_all.sh --lfg checkpoints/lfg_seg_motion_m3n3.pt --kitti360 /data/KITTI-360
#
# Waymo is optional, and evaluated only if you point at it:
#
#   --waymo /data/waymo_v2/validation      the Waymo split directory
#
# The Pi3 baseline needs the upstream pi3 package importable; its weights download on use.
#
# Every model is evaluated at both frame rates. Results land in eval/results/<rate>/ as JSON,
# one file per model and dataset.
set -euo pipefail

LFG=""; KITTI360=""; WAYMO=""; DEVICE="cuda"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lfg)      LFG="$2";      shift 2 ;;
    --kitti360) KITTI360="$2"; shift 2 ;;
    --waymo)    WAYMO="$2";    shift 2 ;;
    --device)   DEVICE="$2";   shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$LFG" || -z "$KITTI360" ]]; then
  sed -n '2,12p' "$0" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

missing () { echo "missing $1: $2" >&2; MISSING=1; }
MISSING=0
[[ -f "$LFG" ]]                  || missing "LFG checkpoint" "$LFG"
[[ -d "$KITTI360" ]]             || missing "KITTI-360 directory" "$KITTI360"
[[ -d "$KITTI360/data_2d_raw" ]] || missing "KITTI-360 images" "$KITTI360/data_2d_raw"
[[ -z "$WAYMO" || -d "$WAYMO/camera_image" ]] || missing "Waymo parquet" "$WAYMO/camera_image"
if [[ "$MISSING" -ne 0 ]]; then
  echo "see the Evaluation section of the README for the expected layout" >&2
  exit 1
fi

run () {   # run <model> <dataset> <data-root> <clip-list> <output-name> [checkpoint]
  local model=$1 dataset=$2 root=$3 clips=$4 name=$5 checkpoint=${6:-}
  local log
  log=$(mktemp)
  printf '%-12s %-9s ' "$model" "$dataset"
  if python evaluate.py --model "$model" --checkpoint "$checkpoint" \
      --dataset "$dataset" --data-root "$root" --clip-list "eval/clips/$clips" \
      --frame-stride "$STRIDE" --device "$DEVICE" \
      --output "eval/results/$RATE/$name.json" >"$log" 2>&1; then
    echo "ok"
  else
    echo "FAILED"
    # surface why: the harness reports missing data and bad arguments on the last line
    grep -vE "^\s*$|Warning|warn" "$log" | tail -3 | sed 's/^/               /' >&2
    FAILURES=$((FAILURES + 1))
  fi
  rm -f "$log"
}

FAILURES=0

for RATE in 10hz 2hz; do
  [[ "$RATE" == "10hz" ]] && STRIDE=1 || STRIDE=5
  mkdir -p "eval/results/$RATE"
  echo
  echo "===== $RATE ====="

  echo "KITTI-360 — depth, semantics and trajectory"
  run lfg        kitti360 "$KITTI360" kitti360_200.txt lfg        "$LFG"
  run vggt       kitti360 "$KITTI360" kitti360_200.txt vggt
  run da3        kitti360 "$KITTI360" kitti360_200.txt da3
  run segformer  kitti360 "$KITTI360" kitti360_200.txt segformer
  run maskformer kitti360 "$KITTI360" kitti360_200.txt maskformer
  run static     kitti360 "$KITTI360" kitti360_200.txt static
  run pi3        kitti360 "$KITTI360" kitti360_200.txt pi3

  if [[ -n "$WAYMO" ]]; then
    echo
    echo "Waymo — depth and trajectory"
    run lfg  waymo "$WAYMO" waymo_200.txt waymo_lfg  "$LFG"
    run vggt waymo "$WAYMO" waymo_200.txt waymo_vggt
    run da3  waymo "$WAYMO" waymo_200.txt waymo_da3
    run pi3  waymo "$WAYMO" waymo_200.txt waymo_pi3
  fi

done

echo
if [[ "$FAILURES" -gt 0 ]]; then
  echo "$FAILURES run(s) failed; results for the rest are in eval/results/" >&2
  exit 1
fi
echo "results written to eval/results/10hz/ and eval/results/2hz/"
