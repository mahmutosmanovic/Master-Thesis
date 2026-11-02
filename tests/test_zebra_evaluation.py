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

    def test_evaluation_reports_actual_steps_separately_from_horizon(self):
        from scripts.eval_zebras import summarize
        rows = [{'reward': 2, 'target_visible_fraction': 1, 'mean_distance_m': 10,
                 'disturbance': 0.2},
                {'reward': 0, 'target_visible_fraction': 0, 'mean_distance_m': 20,
                 'disturbance': 0.4}]
        stats = summarize(rows, horizon_steps=10, dt=0.1)
        self.assertEqual(stats['mean_reward_per_step'], 1)
        self.assertEqual(stats['reward_per_horizon_step'], 0.2)
        self.assertEqual(stats['visible_fraction'], 0.5)
        self.assertEqual(stats['mean_distance_m'], 15)

    def test_rejects_incompatible_checkpoint_layout(self):
        import yaml
        from scripts.eval_zebras import read_run_config
        cfg = {'animal': {'env': {'count': 1}},
               'model': {'space': {'animal_features': 8}}}
        (self.root/'config.yaml').write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, 'older observation layout'):
            read_run_config(self.root)

    def test_run_config_sets_checkpoint_directory_without_editing_saved_config(self):
        import yaml
        from box import Box
        from scripts.eval_zebras import read_run_config
        cfg = {'animal': {'env': {'count': 1}},
               'model': {'space': {'animal_features': 12}},
               'run_dir': '/old/training/machine/run'}
        path = self.root/'config.yaml'
        path.write_text(yaml.safe_dump(cfg))
        before = path.read_bytes()
        loaded = Box(read_run_config(self.root))
        self.assertEqual(loaded.run_dir, str(self.root.resolve()))
        self.assertEqual(path.read_bytes(), before)

    def test_sparse_segments_are_kept_whole_for_random_sampling(self):
        out = self.root/'full'
        report = prepare(self.csv, out, individual='Z3864', episode_seconds=None)
        self.assertEqual(report['segments'], 1)
        self.assertEqual(len(list(out.glob('*.npz'))), 1)
        with np.load(out/'00000000.npz') as z:
            np.testing.assert_allclose(z['t'], [0, 3578, 7167])

    def test_sampling_includes_last_valid_start_and_2049_positions(self):
        from scripts.eval_zebras import sample_replay_window
        class LastStart:
            def integers(self, upper):
                self.upper = upper
                return upper-1
        rng = LastStart()
        np.savez(self.root/'sparse.npz', t=[0., 300.], x=[0., 300.], y=[0., 600.])
        manifest = pd.DataFrame([{'path': 'sparse.npz', 'duration_s': 300.,
                                 'time_scale': 1., 'start_time': '2007-10-25T00:00:00Z'}])
        chosen, index = sample_replay_window(manifest, self.root, self.root/'samples',
                                            0, 2048, 0.1, rng)
        self.assertEqual(rng.upper, 953)
        self.assertEqual(chosen['sample_start_index'].iloc[0], 952)
        self.assertEqual(index, 0)
        with np.load(chosen['path'].iloc[0]) as z:
            self.assertEqual(len(z['t']), 2049)
            np.testing.assert_allclose(z['t'], np.arange(2049)*0.1)
            np.testing.assert_allclose(z['x'], z['t'], atol=1e-10)
            np.testing.assert_allclose(z['y'], 2*z['t'], atol=1e-10)
        a, _ = sample_replay_window(manifest, self.root, self.root/'samples',
                                    1, 2048, 0.1, np.random.default_rng(42))
        b, _ = sample_replay_window(manifest, self.root, self.root/'samples',
                                    2, 2048, 0.1, np.random.default_rng(42))
        self.assertEqual(a['sample_start_index'].iloc[0], b['sample_start_index'].iloc[0])

    def test_sampling_rejects_tracks_shorter_than_episode(self):
        from scripts.eval_zebras import sample_replay_window
        manifest = pd.DataFrame([{'duration_s': 200.}])
        with self.assertRaisesRegex(ValueError, 'No continuous track is long enough'):
            sample_replay_window(manifest, self.root, self.root/'none', 0,
                                 2048, 0.1, np.random.default_rng(42))

    def test_refuses_overwriting_existing_output(self):
        out = self.root/'existing'
        out.mkdir()
        (out/'keep').write_text('existing')
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            prepare(self.csv, out)


if __name__ == '__main__':
    unittest.main()
