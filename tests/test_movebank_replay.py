"""Regression checks for sparse GPS fixes and replay overrides."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from scripts.prepare_movebank import prepare


class MovebankReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / 'zebras.csv'
        pd.DataFrame({
            'individual-local-identifier': ['Z3864'] * 3 + ['other'],
            'timestamp': ['2007-10-25 00:02:11', '2007-10-25 01:01:49',
                          '2007-10-25 02:01:38', '2007-10-25 00:02:11'],
            'location-long': [23.523616, 23.5234968, 23.5234626, 24],
            'location-lat': [-19.3915311, -19.3916838, -19.3916506, -20],
            'visible': [True] * 4,
        }).to_csv(self.csv, index=False)

    def test_hourly_fixes_preserve_timing_distance_and_ground_plane(self):
        out = self.root / 'tracks'
        report = prepare(self.csv, out, individual='Z3864', episode_seconds=4000)
        self.assertEqual(report['episodes'], 2)
        self.assertEqual(report['usable_rows'], 3)
        manifest = pd.read_parquet(out / 'manifest.parquet')
        self.assertEqual(set(manifest['individual-local-identifier']), {'Z3864'})
        transform = Transformer.from_crs(4326, 32734, always_xy=True)
        x, y = transform.transform([23.523616, 23.5234968, 23.5234626],
                                  [-19.3915311, -19.3916838, -19.3916506])
        with np.load(out / manifest.iloc[0]['path']) as z:
            np.testing.assert_allclose(z['t'], [0, 3578, 4000])
            expected_x = np.interp(z['t'], [0, 3578, 7167], x) - x[0]
            expected_y = np.interp(z['t'], [0, 3578, 7167], y) - y[0]
            np.testing.assert_allclose(z['x'], expected_x)
            np.testing.assert_allclose(z['y'], expected_y)
        self.assertAlmostEqual(manifest['duration_s'].sum(), 7167)

    def test_time_compression_is_explicit_and_distance_unchanged(self):
        out = self.root / 'fast'
        report = prepare(self.csv, out, individual='Z3864', episode_seconds=200, time_scale=60)
        self.assertEqual(report['time_scale'], 60)
        with np.load(out / '00000000.npz') as z:
            np.testing.assert_allclose(z['t'], np.array([0, 3578, 7167])/60)

    def test_short_gap_filter_does_not_silently_write_empty_manifest(self):
        with self.assertRaisesRegex(ValueError, 'No replayable segments'):
            prepare(self.csv, self.root/'empty', individual='Z3864', max_gap=20)

    def test_duplicates_and_invalid_fixes_are_filtered(self):
        df = pd.read_csv(self.csv)
        df = pd.concat([df, df.iloc[[0]], df.iloc[[1]].assign(visible=False)])
        df.to_csv(self.csv, index=False)
        report = prepare(self.csv, self.root/'clean', individual='Z3864')
        self.assertEqual(report['usable_rows'], 3)

    def test_replay_moves_animal_to_interpolated_positions(self):
        from environment.entity import Animal
        from environment.behaviors import REPLAY_CFG
        from environment.immutables import MovementDim
        out = self.root/'replay'
        prepare(self.csv, out, individual='Z3864', episode_seconds=4000)
        animal = Animal({'dt': 0.1, 'animal': {'init': {
            'min_speed': 0, 'max_speed': 12, 'movement_dim': MovementDim.TWO_D,
            'behavior': REPLAY_CFG(str(out/'manifest.parquet'))}}})
        animal.behavior.reset(animal, np.random.default_rng(42))
        for step in range(100):
            animal.update_vel(np.random.default_rng(42))
            animal.update_pos()
        with np.load(out/'00000000.npz') as z:
            expected = [np.interp(10, z['t'], z['x']), np.interp(10, z['t'], z['y']), 0]
        np.testing.assert_allclose(animal.pos.to_numpy(), expected, atol=1e-7)
        self.assertFalse(animal.behavior.can_flee)

    def test_run_override_preserves_training_file_and_model_settings(self):
        from unittest.mock import patch
        import yaml
        from scripts.run_utils import load_run
        out = self.root/'override'
        prepare(self.csv, out, individual='Z3864')
        config_file = self.root/'config.yaml'
        saved = {'animal': {'init': {'behavior': 'CRW_CFG', 'movement_dim': 'TWO_D'}},
                 'model': {'space': {'animal_features': 12}},
                 'behavior_cfg': 'missing-original-behavior-directory'}
        config_file.write_text(yaml.safe_dump(saved))
        before = config_file.read_bytes()
        with patch('scripts.run_utils.resolve_run_dir', return_value=self.root):
            cfg, _ = load_run('test', replay_manifest=out/'manifest.parquet')
        self.assertEqual(config_file.read_bytes(), before)
        self.assertEqual(cfg['model'], saved['model'])
        self.assertEqual(cfg['animal']['init']['behavior'].manifest_path,
                         str(out/'manifest.parquet'))

    def test_refuses_overwriting_existing_output(self):
        out = self.root/'existing'
        out.mkdir()
        (out/'keep').write_text('existing')
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            prepare(self.csv, out)


if __name__ == '__main__':
    unittest.main()
