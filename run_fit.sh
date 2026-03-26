#!/usr/bin/env bash
set -euo pipefail

manifests=(
  "pigeons"
  "jackals"
  "spur_winged_lapwings"
)

poi_inferences=(
  "km_sm"
)

for animal in "${manifests[@]}"; do
  for poi_inf in "${poi_inferences[@]}"; do
    outdir="./behaviors_t3/${animal}_${poi_inf}"

    case "$animal" in
      pigeons)
        arrive_dist=15
        bias_gain=0.2
        poi_eps=5
        ;;
      jackals)
        arrive_dist=50
        bias_gain=0.2
        poi_eps=75
        ;;
      spur_winged_lapwings)
        arrive_dist=50
        bias_gain=0.2
        poi_eps=75
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
      --poi_bias_gain "$bias_gain" \
      --poi_eps "$poi_eps"
  done
done