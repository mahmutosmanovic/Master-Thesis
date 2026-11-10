import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eval_zebras import check_fix_capacity, prepare_sequential
from scripts.prepare_movebank import read_fixes


def fixes(n, identifier='zebra'):
    return pd.DataFrame({
        'individual-local-identifier': [identifier] * n,
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='15min', tz='UTC'),
        'location-long': 24 + np.arange(n) * 1e-6,
        'location-lat': np.full(n, -20.0),
    })


class SequentialZebraTests(unittest.TestCase):
    def test_capacity_uses_time_and_exact_boundary(self):
        col = 'individual-local-identifier'
        source = fixes(2)
        source['timestamp'] = pd.to_datetime(['2020-01-01T00:00:00Z',
                                             '2020-01-01T00:13:39.200Z'], format='mixed')
        self.assertEqual(check_fix_capacity(source, col, 4, 2048, 0.1), 4)
        with self.assertRaisesRegex(ValueError, 'Not enough recorded duration'):
            check_fix_capacity(source, col, 5, 2048, 0.1)
        # Many more than five episodes fit in 9,000 hourly/quarter-hourly fixes.
        self.assertGreater(check_fix_capacity(fixes(9000), col, 5, 2048, 0.1), 5)
        with self.assertRaises(ValueError):
            check_fix_capacity(source, col, 1, 2048, 0)

    def test_hourly_fixes_interpolate_at_tick_and_windows_are_consecutive(self):
        from pyproj import Transformer
        source = fixes(3)
        source['timestamp'] = pd.to_datetime([
            '2007-10-25T00:02:11Z', '2007-10-25T01:01:49Z',
            '2007-10-25T02:01:38Z'])
        source['location-long'] = [23.523616, 23.5234968, 23.5234626]
        source['location-lat'] = [-19.3915311, -19.3916838, -19.3916506]
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = prepare_sequential(
                source, 'individual-local-identifier', directory, 20, 2048, 0.1)
            x, y = Transformer.from_crs(4326, int(manifest.iloc[0].utm_epsg),
                                      always_xy=True).transform(
                source['location-long'].to_numpy(), source['location-lat'].to_numpy())
            times = np.array([0, 3578, 7167])
            for episode, row in manifest.iterrows():
                start = episode * 204.8
                query = start + np.arange(2049) * 0.1
                expected_x = np.interp(query, times, x - x[0])
                expected_y = np.interp(query, times, y - y[0])
                with np.load(row.path) as track:
                    np.testing.assert_allclose(track['x'], expected_x - expected_x[0], atol=1e-8)
                    np.testing.assert_allclose(track['y'], expected_y - expected_y[0], atol=1e-8)
                    np.testing.assert_allclose(track['t'], np.arange(2049) * 0.1)
                    if episode == 0:
                        # Tick speed equals timestamp-based interval speed, not displacement / dt.
                        speed = np.hypot(track['x'][1], track['y'][1]) / 0.1
                        self.assertAlmostEqual(speed, np.hypot(x[1]-x[0], y[1]-y[0])/3578)
                self.assertEqual(row.start_time, source.timestamp.iloc[0] + pd.to_timedelta(start, unit='s'))
                if episode:
                    self.assertEqual(row.start_time, manifest.iloc[episode-1].end_time)

    def test_failure_creates_no_outputs_and_animals_are_not_joined(self):
        source = pd.concat([fixes(2, 'a'), fixes(2, 'b')])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'unused'
            with self.assertRaises(ValueError):
                # Each animal spans 900 seconds: neither fits this 1,000-second episode.
                prepare_sequential(source, 'individual-local-identifier', out, 1, 10000, 0.1)
            self.assertFalse(out.exists())

    def test_cleanup_before_counting(self):
        source = fixes(4)
        source = pd.concat([source.iloc[::-1], source.iloc[[0]]], ignore_index=True)
        source.loc[0, 'location-lat'] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / 'fixes.csv'
            source.to_csv(csv, index=False)
            cleaned, _, original = read_fixes(csv)
            self.assertEqual(original, 5)
            self.assertEqual(len(cleaned), 3)
            self.assertTrue(cleaned.timestamp.is_monotonic_increasing)


if __name__ == '__main__':
    unittest.main()
