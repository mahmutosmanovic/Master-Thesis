"""Regression checks for individual selection, time units and checkpoint paths."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.baboon_data import select_fixes
from scripts.eval_baboons import resolve_model
from scripts.eval_zebras import check_fix_capacity, prepare_sequential


def fixes(identifier, n, start='2012-08-01 02:59:57'):
    return pd.DataFrame({
        'timestamp': pd.date_range(start, periods=n, freq='s'),
        'location-long': 36.9229031 + np.arange(n) * 1e-6,
        'location-lat': np.full(n, 0.3509073),
        'individual-local-identifier': [identifier] * n,
    })


class BaboonEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root/'baboons.csv'

    def write(self, *frames):
        pd.concat(frames, ignore_index=True).to_csv(self.csv, index=False)

    def test_select_one_eligible_id_and_first_2048_chronological_fixes(self):
        animal = fixes('1', 2055)
        self.write(fixes('short', 100), animal.iloc[::-1], fixes('2', 2100))
        selected, col, report = select_fixes(self.csv)
        self.assertEqual(set(selected[col]), {'1'})
        self.assertEqual(len(selected), 2048)
        self.assertEqual(selected.timestamp.iloc[-1], pd.Timestamp('2012-08-01 03:34:04', tz='UTC'))
        self.assertEqual(report['duration_seconds'], 2047)
        self.assertEqual(report['median_fix_interval_seconds'], 1)

    def test_cleanup_and_explicit_individual(self):
        animal = fixes('001', 2049)
        bad = animal.iloc[[0]].assign(**{'location-long': np.inf})
        self.write(fixes('2', 2050), bad, animal, animal.iloc[[0]])
        selected, col, _ = select_fixes(self.csv, '001')
        self.assertEqual(set(selected[col]), {'001'})
        self.assertEqual(len(selected), 2048)
        self.assertTrue(np.isfinite(selected['location-long']).all())
        with self.assertRaisesRegex(ValueError, 'No eligible baboon'):
            select_fixes(self.csv, 'absent')

    def test_no_joining_short_individuals_or_skipping_initial_gap(self):
        self.write(fixes('1', 1100), fixes('2', 1100))
        with self.assertRaisesRegex(ValueError, 'only 1100'):
            select_fixes(self.csv)
        animal = fixes('1', 5000)
        animal.loc[1:, 'timestamp'] += pd.Timedelta(seconds=60)
        self.write(animal)
        with self.assertRaisesRegex(ValueError, '61s gap'):
            select_fixes(self.csv)

    def test_chunk_boundary_keeps_earliest_fixes_and_deduplicates(self):
        # Force the normal chunked reader through multiple chunks cheaply.
        from unittest.mock import patch
        first, second = fixes('1', 2050).iloc[1000:], fixes('1', 2050).iloc[:1001]
        self.write(first, second)
        original = pd.read_csv
        def small_chunks(*args, **kwargs):
            if 'chunksize' in kwargs:
                kwargs['chunksize'] = 500
            return original(*args, **kwargs)
        with patch('scripts.baboon_data.pd.read_csv', side_effect=small_chunks):
            selected, _, report = select_fixes(self.csv)
        self.assertEqual(report['input_rows'], 2051)
        self.assertEqual(report['duration_seconds'], 2047)
        self.assertEqual(len(selected), 2048)

    def test_replay_preserves_seconds_and_utm_metres(self):
        self.write(fixes('1', 2048))
        selected, col, _ = select_fixes(self.csv)
        self.assertEqual(check_fix_capacity(selected, col, 1, 2048, .1), 9)
        manifest, _ = prepare_sequential(selected, col, self.root/'tracks', 1, 2048, .1)
        row = manifest.iloc[0]
        self.assertEqual(row.utm_epsg, 32637)
        with np.load(row.path) as data:
            self.assertEqual(len(data['t']), 2049)  # reset plus 2048 steps
            self.assertAlmostEqual(data['t'][-1], 204.8)
            # Longitude increment is about 0.111m/s here, not 1.11m/s.
            self.assertAlmostEqual(data['x'][10], .111, delta=.001)
            self.assertAlmostEqual(data['x'][1], data['x'][10]/10, places=7)

    def test_resolve_exact_checkpoint_and_run_directory(self):
        run = self.root/'run'
        run.mkdir()
        with (run/'config.yaml').open('w') as f:
            yaml.safe_dump({'agent_type': 'ppo'}, f)
        checkpoint = run/'ppo_best.pt'
        checkpoint.touch()
        self.assertEqual(resolve_model(run=str(run)), (run, checkpoint))
        self.assertEqual(resolve_model(model_path=str(checkpoint)), (run, checkpoint))
        with self.assertRaisesRegex(FileNotFoundError, 'Checkpoint not found'):
            resolve_model(run=str(run), weights='last')


if __name__ == '__main__':
    unittest.main()
