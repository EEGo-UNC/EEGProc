"""Small signal/label contract checks; no training or dataset required."""
import unittest
import numpy as np
from preprocess import LABELS, preprocess_trial


class AdapterTests(unittest.TestCase):
    def test_session_labels(self):
        for labels in LABELS:
            self.assertEqual(np.bincount(labels).tolist(), [6,6,6,6])

    def test_window_shape_channel_band_order_and_normalization(self):
        signal = np.random.default_rng(2).normal(size=(62,1601))
        windows=preprocess_trial(signal)
        self.assertEqual(windows.shape,(2,800,310))
        np.testing.assert_allclose(np.mean(windows**2,axis=(1,2)),1,rtol=1e-5)
        # CAR is linear: the sum over channels must remain near zero in
        # every band; this also verifies the channel-major layout.
        grouped=windows.reshape(2,800,62,5)
        self.assertLess(float(np.max(np.abs(grouped.sum(axis=2)))),1e-4)

    def test_bad_channels_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_trial(np.zeros((14,800)))


if __name__=='__main__':
    unittest.main()
