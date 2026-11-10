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
    def test_capacity_and_exact_boundary(self):
        col = 'individual-local-identifier'
        self.assertEqual(check_fix_capacity(fixes(9000), col, 4, 2048), 4)
        with self.assertRaisesRegex(ValueError, 'Not enough usable GPS points'):
            check_fix_capacity(fixes(9000), col, 5, 2048)
        self.assertEqual(check_fix_capacity(fixes(2049), col, 1, 2048), 1)
        with self.assertRaises(ValueError):
            check_fix_capacity(fixes(2048), col, 1, 2048)

    def test_order_boundaries_and_recorded_positions(self):
        from pyproj import Transformer
        source = fixes(9000)
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = prepare_sequential(
                source, 'individual-local-identifier', directory, 4, 2048, 0.1)
            self.assertEqual(manifest.source_start_index.tolist(), [0, 2048, 4096, 6144])
            self.assertEqual(manifest.source_end_index.tolist(), [2048, 4096, 6144, 8192])
            for episode, row in manifest.iterrows():
                block = source.iloc[episode * 2048:episode * 2048 + 2049]
                x, y = Transformer.from_crs(4326, int(row.utm_epsg), always_xy=True).transform(
                    block['location-long'].to_numpy(), block['location-lat'].to_numpy())
                with np.load(row.path) as track:
                    np.testing.assert_allclose(track['x'], x - x[0])
                    np.testing.assert_allclose(track['y'], y - y[0])
                    np.testing.assert_allclose(track['t'], np.arange(2049) * 0.1)
                self.assertEqual(row.start_time, block.timestamp.iloc[0])
                self.assertEqual(row.end_time, block.timestamp.iloc[-1])

    def test_failure_creates_no_outputs_and_animals_are_not_joined(self):
        source = pd.concat([fixes(2000, 'a'), fixes(2000, 'b')])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'unused'
            with self.assertRaises(ValueError):
                prepare_sequential(source, 'individual-local-identifier', out, 1, 2048, 0.1)
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
