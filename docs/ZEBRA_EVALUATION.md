# Zebra GPS evaluation

From the project root, with the normal training dependencies installed:

```bash
python3 -m pip install pyarrow
python3 -m scripts.eval_zebras --csv data/zebras/Z3864.csv --run YOUR_RUN --episodes 10
```

`YOUR_RUN` is an existing folder under `runs/`; `latest` is also accepted. The
command uses `best` weights; use `--weights last` for the last checkpoint. No
baseline is evaluated. Your training config and source CSV are not modified.

To export a video and snapshots of the first episode, append `--record` (requires
ffmpeg on PATH). For another individual's file, change `--csv`, e.g.
`--csv data/zebras/Z3743.csv`. With a multi-individual CSV, add
`--individual Z3864` or omit it to prepare all individuals separately.

Inspect the data without loading any policy:

```bash
python3 -m scripts.eval_zebras --csv data/zebras/Z3864.csv --prepare-only
```

A new directory under `evals/zebras/` contains:

- `tracks/report.json`: fix counts, sampling intervals and recorded step speeds.
- `tracks/trajectories.png`: original fixes connected by interpolation lines.
- `tracks/manifest.csv` and `.parquet`: replay windows with source dates and IDs.
- `steps.csv`: reward, visibility, disturbance, animal/drone coordinates and distances.
- `episodes.csv`: episode statistics, durations, track loss and selected window.
- `summary.json`: aggregate performance, checkpoint, source CSV and time scale.
- `performance.png` and `drone_trajectory.png`: performance and tracking plots.
- `config.yaml`: evaluation config snapshot.
- `recording/`: optional first-episode video and snapshots.

Use `--out evals/zebras/my_test` to set a new or empty output directory. Preparation
alone creates only the `tracks/` outputs. Each evaluation selects at most
`--episodes` distinct windows, evenly spaced through the manifest, with no repeated
windows. Seeds control drone spawning; short final windows and early track loss
can result in different episode lengths. Mean reward per executed step and reward
per configured horizon step are both reported. Aggregate visibility, distance and
disturbance are weighted by executed steps. Early termination should be assessed
alongside episode duration and track-loss count.

The trained run must use one animal and the observation layout supported by the
current environment (12 animal features). Drone count/settings and model dimensions
are preserved. Older 8-feature checkpoints require their matching environment or
retraining; changing the feature count in YAML cannot convert a checkpoint.

CSV fields: `timestamp`, `location-long`, `location-lat`, plus
`individual-local-identifier` or `tag-local-identifier`. Timestamps are interpreted
as UTC. Coordinates are projected from latitude/longitude to a UTM CRS per segment
and translated to local metres. Invalid/duplicate/outlier/invisible fixes are
filtered; time gaps exceeding `--max-gap 7200` original seconds split segments.
No smoothing or spatial scaling is applied. Inspect the report/plot for remaining
implausible jumps. Prepared windows match the run's `dt * max_episode_steps`.

The example fixes are about one hour apart. Linear interpolation cannot recover
unobserved short-term turns, and 205-second windows may contain only a small
straight portion. For a visibly faster demonstration, use `--time-scale 60`; this
multiplies speeds by 60 and must be labelled as time-compressed. Default `1`
preserves recorded time. Playback is fixed-path and disables fleeing: disturbance
is a model score, not validation of real zebra responses to a drone. Neighboring
windows from the same animal are not independent biological samples.
