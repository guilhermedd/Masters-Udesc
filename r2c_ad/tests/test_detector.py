import unittest

import numpy as np
import torch

from r2c.alarms import calibrate, decisions, starts
from r2c.data import Windows
from r2c.model import RelationalDetector


class DetectorTests(unittest.TestCase):
    def test_target_hidden_in_conditional_head(self):
        torch.manual_seed(11)
        model = RelationalDetector(3, 8, 4, 8, 1).eval()
        x = torch.randn(2, 8, 3)
        y = torch.randn(2, 3)
        mask = torch.ones_like(y, dtype=torch.bool)
        with torch.no_grad():
            h = model.history(x, torch.ones_like(x, dtype=torch.bool))
            before = model.predict(h, y, mask, model.edges)
            y[:, 1] += 100
            after = model.predict(h, y, mask, model.edges)
        torch.testing.assert_close(before[:, 1], after[:, 1])

    def test_window_ends_before_target(self):
        x = np.arange(24, dtype=np.float32).reshape(12, 2)
        x[1, 0] = np.nan
        data = Windows(x, 4, np.zeros(2, np.float32), np.ones(2, np.float32))
        history, observed, target, _ = data[0]
        self.assertFalse(observed[1, 0])
        self.assertEqual(history[1, 0], 0)
        np.testing.assert_equal(history[-1].numpy(), x[3])
        np.testing.assert_equal(target.numpy(), x[4])

    def test_joint_policy_budget_and_missing_reset(self):
        series = np.tile([0.1, 0.5, 0.92, 0.94, np.nan], 30)
        high, events = calibrate(series, 5)
        active = decisions(series, high)
        self.assertFalse(active[~np.isfinite(series)].any())
        self.assertEqual(starts(active).sum(), events)
        self.assertLessEqual(1000 * events / np.isfinite(series).sum(), 5)

    def test_alarm_at_start_counts_once(self):
        np.testing.assert_equal(starts(np.array([True, True, False, True])), [1, 0, 0, 1])
