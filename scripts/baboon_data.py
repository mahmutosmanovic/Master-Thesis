"""Select one baboon's earliest 2,048 valid, distinct GPS fixes."""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path('data/baboons/Collective movement in wild baboons (data from Strandburg-Peshkin et al. 2015).csv')
SOURCE_POINTS = 2048
ID_COL = 'individual-local-identifier'
REQUIRED = [ID_COL, 'timestamp', 'location-long', 'location-lat']


def select_fixes(csv_path, individual=None, max_gap=2.0):
    """Scan in chunks, retaining only the earliest 2048 fixes per candidate ID.

    Automatic selection uses first appearance in the CSV. Invalid/duplicate
    fixes are removed before counting. Never join individuals, skip ahead to a
    later window, compress timestamps, or interpolate across a long gap.
    """
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError('max_gap must be finite and positive')
    csv_path = Path(csv_path)
    columns = pd.read_csv(csv_path, nrows=0).columns
    missing = [column for column in REQUIRED if column not in columns]
    if missing:
        raise ValueError(f'Missing CSV columns: {missing}')
    usecols = REQUIRED + [c for c in ['visible', 'manually-marked-outlier'] if c in columns]
    candidates = {}
    input_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, dtype={ID_COL: str}, chunksize=200_000):
        input_rows += len(chunk)
        if individual is not None:
            chunk = chunk[chunk[ID_COL] == str(individual)].copy()
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', utc=True)
        for col in ['location-long', 'location-lat']:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk.dropna(subset=REQUIRED)
        chunk = chunk[chunk['location-long'].between(-180, 180) &
                      chunk['location-lat'].between(-80, 84)]
        if 'visible' in chunk:
            chunk = chunk[chunk['visible'].astype(str).str.lower().isin(['true', '1'])]
        if 'manually-marked-outlier' in chunk:
            chunk = chunk[~chunk['manually-marked-outlier'].astype(str).str.lower().isin(['true', '1'])]
        for identifier, rows in chunk.groupby(ID_COL, sort=False):
            previous = candidates.get(identifier)
            if previous is not None:
                rows = pd.concat([previous, rows], ignore_index=True)
            rows = rows.sort_values('timestamp', kind='stable').drop_duplicates('timestamp')
            candidates[identifier] = rows.head(SOURCE_POINTS).copy()

    rejected = {}
    for identifier, rows in candidates.items():
        if len(rows) < SOURCE_POINTS:
            rejected[identifier] = f'only {len(rows)} usable fixes'
            continue
        gaps = rows['timestamp'].diff().dt.total_seconds().dropna()
        if gaps.max() > max_gap:
            rejected[identifier] = f'first {SOURCE_POINTS} fixes contain a {gaps.max():g}s gap (limit {max_gap:g}s)'
            continue
        rows = rows.reset_index(drop=True)
        return rows, ID_COL, {
            'dataset': 'Collective movement in wild baboons (Strandburg-Peshkin et al. 2015)',
            'source_file': str(csv_path.resolve()),
            'input_rows': input_rows, 'individual': str(identifier),
            'individuals': 1, 'usable_rows': len(rows), 'source_points': SOURCE_POINTS,
            'selection': 'first 2048 cleaned, timestamp-sorted unique fixes of one individual',
            'start_time': str(rows['timestamp'].iloc[0]),
            'end_time': str(rows['timestamp'].iloc[-1]),
            'duration_seconds': float((rows['timestamp'].iloc[-1]-rows['timestamp'].iloc[0]).total_seconds()),
            'median_fix_interval_seconds': float(gaps.median()),
            'max_fix_interval_seconds': float(gaps.max()), 'max_allowed_gap_seconds': max_gap,
            'replay_timing': 'gps_timestamp_aligned_linear_interpolation',
            'rejected_individuals': rejected,
        }
    detail = '; '.join(f'{key}: {value}' for key, value in rejected.items())
    raise ValueError(f'No eligible baboon with {SOURCE_POINTS} fixes. {detail or "No matching valid individual."}')
