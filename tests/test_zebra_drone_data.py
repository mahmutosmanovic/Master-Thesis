import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from scripts.zebra_drone_data import read_drone_tracks, fetch_source
from scripts.eval_zebras import prepare_sequential


class DroneZebraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'source.h5'
        # Head-tail length is 10 source units; motion is 1 source unit/frame.
        self.tracks = np.zeros((1, 2, 2, 100))
        self.tracks[0, 0, 0] = np.arange(100) + 10
        self.tracks[0, 0, 1] = np.arange(100)
        self.save()

    def save(self):
        with h5py.File(self.path, 'w') as f:
            f['tracks'] = self.tracks
            f['node_names'] = [b'H', b'T']
            f['track_names'] = [b'track_0']

    def test_scale_and_frame_timing(self):
        fixes, col, report = read_drone_tracks(self.path, body_length_m=2)
        self.assertAlmostEqual(report['metres_per_source_unit'], .2)
        np.testing.assert_allclose(fixes.x_m, (np.arange(100)+5)*.2)
        self.assertAlmostEqual(fixes.source_time_s.iloc[30], 30/29.97)
        out = Path(self.temp.name)/'out'
        manifest, available = prepare_sequential(fixes, col, out, 3, 10, .1)
        self.assertEqual(available, 3)
        self.assertEqual(manifest.source_start_seconds.tolist(), [0, 1, 2])
        with np.load(manifest.iloc[0].path) as track:
            np.testing.assert_allclose(track['x'], np.arange(11)*.1*29.97*.2, atol=1e-7)
            np.testing.assert_allclose(track['y'], 0)

    def test_missing_frames_keep_time_and_long_gaps_split(self):
        self.tracks[:, :, :, 10:12] = np.nan  # short gap
        self.tracks[:, :, :, 40:65] = np.nan  # > .5 s
        self.save()
        fixes, col, _ = read_drone_tracks(self.path, max_gap=.5)
        self.assertEqual(fixes[col].nunique(), 2)
        row = fixes[fixes.source_frame == 12].iloc[0]
        self.assertAlmostEqual(row.source_time_s, 12/29.97)
        out = Path(self.temp.name)/'too_many'
        with self.assertRaisesRegex(ValueError, 'Not enough recorded duration'):
            prepare_sequential(fixes, col, out, 1, 20, .1)
        self.assertFalse(out.exists())

    def test_invalid_individual_and_scale(self):
        with self.assertRaisesRegex(ValueError, 'Unknown zebra'):
            read_drone_tracks(self.path, individual='missing')
        with self.assertRaises(ValueError):
            read_drone_tracks(self.path, body_length_m=0)

    def test_corrupt_cache_rejected_without_network(self):
        with patch('scripts.zebra_drone_data.urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                fetch_source(self.path)
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
