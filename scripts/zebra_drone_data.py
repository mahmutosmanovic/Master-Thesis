"""Load time-preserving, ground-plane zebra tracks from Duporge et al. (2025).

The cleaned SLEAP HDF5 retains missing frames and head/tailbase keypoints.
The visualization CSV exports do not retain frame indices, so are not used.
"""
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

SOURCE_COMMIT = "d4c36aebf36da43c677a53a044925e5774919096"
SOURCE_PATH = ("data/approach-sfm-interp/"
               "20250325_2228_id_sfm_interp_PCS_2d_20250516_155745_clean.h5")
SOURCE_URL = ("https://raw.githubusercontent.com/neuroinformatics-unit/"
              f"zebras-stitching/{SOURCE_COMMIT}/{SOURCE_PATH}")
SOURCE_SHA256 = "a7ca88c70eb761e14b3b70b4f0290c6cc9e81c32255c2d7dae2bb28a8ba91334"
DEFAULT_FILE = Path("data/zebras/drone_escape_clean.h5")
FPS = 29.97


def fetch_source(path=DEFAULT_FILE):
    """Cache a pinned source file; verify bytes before use or replacement."""
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"Source checksum mismatch: {path}. Remove the file and retry.")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(path)
    return path.resolve()


def read_drone_tracks(path, individual=None, body_length_m=2.0, max_gap=0.5):
    """Return metric planar centroids, preserving source frame times and gaps.

    A global median head-to-tailbase distance supplies the scale. Both keypoints
    must be finite. Only gaps <= max_gap seconds may be interpolated; larger
    gaps split tracks. No extrapolation, endpoint padding, or speed clipping.
    """
    import h5py
    for name, value in [("body_length_m", body_length_m), ("max_gap", max_gap)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    with h5py.File(path) as source:
        tracks = source["tracks"][:].astype(np.float64)
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in source["track_names"][:]]
        nodes = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in source["node_names"][:]]
    if (tracks.ndim != 4 or tracks.shape[:3] != (len(names), 2, len(nodes))
            or set(nodes) != {"H", "T"}):
        raise ValueError("Expected cleaned SLEAP tracks shaped (individual, xy, H/T, frame)")
    if individual is not None and individual not in names:
        raise ValueError(f"Unknown zebra {individual!r}; available: {', '.join(names)}")
    lengths = np.linalg.norm(tracks[:, :, nodes.index("H"), :] -
                             tracks[:, :, nodes.index("T"), :], axis=1)
    valid_lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    if not len(valid_lengths):
        raise ValueError("No finite, positive body lengths for metric calibration")
    median_length = float(np.median(valid_lengths))
    scale = body_length_m / median_length
    groups = []
    input_rows = usable_rows = 0
    for i, name in enumerate(names):
        if individual is not None and name != individual:
            continue
        input_rows += tracks.shape[-1]
        valid = np.isfinite(tracks[i]).all(axis=(0, 1))
        indices = np.flatnonzero(valid)
        usable_rows += len(indices)
        if len(indices) < 2:
            continue
        # Retain original frame indices: missing frames must not compress time.
        parts = np.split(indices, np.flatnonzero(np.diff(indices)/FPS > max_gap) + 1)
        for segment, frames in enumerate(parts):
            if len(frames) < 2:
                continue
            xy = tracks[i][:, :, frames].mean(axis=1).T * scale
            groups.append(pd.DataFrame({
                "replay_track_id": f"{name}/segment_{segment}",
                "individual": name, "source_frame": frames,
                "source_time_s": frames/FPS,
                # Relative video time, not a real recording date.
                "timestamp": pd.to_datetime(frames/FPS, unit="s", utc=True),
                "x_m": xy[:, 0], "y_m": xy[:, 1],
            }))
    if not groups:
        raise ValueError("No valid continuous zebra track segments")
    fixes = pd.concat(groups, ignore_index=True)
    report = {
        "dataset": "Duporge et al. (2025), Tracking the Flight; cleaned SfM ground-plane tracks",
        "source_repository": "https://github.com/neuroinformatics-unit/zebras-stitching",
        "source_commit": SOURCE_COMMIT, "source_url": SOURCE_URL,
        "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "source_fps": FPS, "input_rows": input_rows, "usable_rows": usable_rows,
        "replayable_rows": len(fixes), "individuals": int(fixes.individual.nunique()),
        "segments": int(fixes.replay_track_id.nunique()),
        "assumed_head_to_tailbase_length_m": body_length_m,
        "median_body_length_source_units": median_length,
        "metres_per_source_unit": scale,
        "max_interpolated_gap_s": max_gap,
        "time_origin": "video start; timestamps are relative, not calendar dates",
        "coordinate_system": "local ground-plane xy in assumed metres; z=0",
        "behaviour": "escape response during simulated predation; not undisturbed movement",
        "scale_status": "body-length assumption, not surveyed metric ground truth",
    }
    return fixes, "replay_track_id", report
