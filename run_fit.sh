#!/usr/bin/env bash
set -euo pipefail

manifests=(
  "pigeons"
  "jackals"
  "spur_winged_lapwings"
)

poi_inferences=(
  "km"
  "km_sm"
  "hmm"
)

for animal in "${manifests[@]}"; do
  for poi_inf in "${poi_inferences[@]}"; do
    outdir="./behaviors_40k100/${animal}_${poi_inf}"

    case "$animal" in
      pigeons)
        arrive_dist=25
        bias_gain=0.2
        ;;
      jackals)
        arrive_dist=75
        bias_gain=0.2
        ;;
      spur_winged_lapwings)
        arrive_dist=75
        bias_gain=0.2
        ;;
      *)
        echo "Unknown animal: $animal" >&2
        exit 1
        ;;
    esac

    echo "Running: animal=$animal poi_inference=$poi_inf outdir=$outdir"

    python -m scripts.gps_fit \
      --manifest "./track_segments/${animal}/manifest.parquet" \
      --outdir "$outdir" \
      --poi_inference "$poi_inf" \
      --n_steps 40000 \
      --n_seeds 100 \
      --poi_arrive_dist "$arrive_dist" \
      --poi_bias_gain "$bias_gain"
  done
done