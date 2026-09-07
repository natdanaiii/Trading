import unittest

import numpy as np
import pandas as pd

from dynamic_grid_v1 import (
    build_dynamic_regime,
    round_to_gap,
    run_dynamic_grid_backtest,
)


class TestDynamicGridV1(unittest.TestCase):
    def test_round_to_gap(self):
        self.assertEqual(round_to_gap(111_499, 1000), 111_000)
        self.assertEqual(round_to_gap(111_500, 1000), 112_000)

    def test_regime_is_centered_and_fixed_size(self):
        regime = build_dynamic_regime(
            reference_price=100_000,
            gap=1000,
            number_of_grids=30,
            capital=3000,
        )
        self.assertEqual(regime["floor"], 85_000)
        self.assertEqual(regime["ceiling"], 115_000)
        self.assertEqual(len(regime["buy_prices"]), 30)
        self.assertAlmostEqual(regime["capital_per_grid"], 100.0)
        self.assertTrue(np.allclose(np.diff(regime["buy_prices"]), 1000.0))

    def test_recenter_uses_close_and_is_causal(self):
        df = pd.DataFrame({
            "open_time": pd.date_range(
                "2024-01-01", periods=3, freq="min", tz="UTC"
            ),
            "open": [100.0, 100.0, 111.0],
            "high": [100.0, 111.0, 111.0],
            "low": [100.0, 100.0, 110.0],
            "close": [100.0, 111.0, 110.0],
        })

        result = run_dynamic_grid_backtest(
            df,
            initial_capital=400.0,
            gap=10.0,
            number_of_grids=4,
            recenter_trigger_grids=1,
            buy_fee=0.001,
            sell_fee=0.001,
        )

        log = result["recenter_log"]
        self.assertEqual(len(log), 1)
        self.assertEqual(log.iloc[0]["old_reference"], 100.0)
        self.assertEqual(log.iloc[0]["new_reference"], 110.0)
        self.assertEqual(
            result["equity_curve"].iloc[1]["reference_price"],
            100.0,
        )
        self.assertEqual(
            result["equity_curve"].iloc[2]["reference_price"],
            110.0,
        )

    def test_old_position_survives_recenter_and_keeps_target(self):
        df = pd.DataFrame({
            "open_time": pd.date_range(
                "2024-01-01", periods=2, freq="min", tz="UTC"
            ),
            "open": [100.0, 90.0],
            "high": [100.0, 100.0],
            "low": [89.0, 89.0],
            "close": [90.0, 90.0],
        })

        result = run_dynamic_grid_backtest(
            df,
            initial_capital=400.0,
            gap=10.0,
            number_of_grids=4,
            recenter_trigger_grids=1,
            buy_fee=0.001,
            sell_fee=0.001,
        )

        completed = result["completed_trades"]
        self.assertGreaterEqual(len(result["recenter_log"]), 1)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed.iloc[0]["regime_id"], 0)
        self.assertEqual(completed.iloc[0]["buy_price"], 90.0)
        self.assertEqual(completed.iloc[0]["sell_price"], 100.0)

    def test_cash_never_goes_negative(self):
        df = pd.DataFrame({
            "open_time": pd.date_range(
                "2024-01-01", periods=4, freq="min", tz="UTC"
            ),
            "open": [100.0, 100.0, 90.0, 80.0],
            "high": [100.0, 100.0, 90.0, 80.0],
            "low": [100.0, 89.0, 79.0, 69.0],
            "close": [100.0, 90.0, 80.0, 70.0],
        })

        result = run_dynamic_grid_backtest(
            df,
            initial_capital=400.0,
            gap=10.0,
            number_of_grids=4,
            recenter_trigger_grids=1,
            buy_fee=0.001,
            sell_fee=0.001,
        )

        self.assertGreaterEqual(result["equity_curve"]["cash"].min(), -1e-9)


if __name__ == "__main__":
    unittest.main()
