import ast
import bisect
import heapq
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "Grid_trading.ipynb"


def load_notebook_function(function_name):
    """Load one function definition directly from Grid_trading.ipynb.

    Only the requested FunctionDef node is executed. This avoids running
    Google Drive mounting, historical-data loading, or the full backtest.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    namespace = {
        "np": np,
        "pd": pd,
        "heapq": heapq,
        "bisect": bisect,
    }

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if f"def {function_name}(" not in source:
            continue

        tree = ast.parse(source)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                module = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(module)
                exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
                return namespace[function_name]

    raise RuntimeError(
        f"Function {function_name!r} was not found in {NOTEBOOK_PATH.name}."
    )


build_excel_grid_table = load_notebook_function("build_excel_grid_table")
run_grid_backtest = load_notebook_function("run_grid_backtest")


def make_candles(rows):
    """Build deterministic 1-minute OHLC candles for synthetic tests.

    rows: iterable of (open, high, low, close)
    """
    rows = list(rows)
    times = pd.date_range(
        "2024-01-01",
        periods=len(rows),
        freq="min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "open_time": times,
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
        }
    )


class TestGridTradingEngine(unittest.TestCase):
    """Unit tests using small synthetic price paths with known outcomes."""

    def setUp(self):
        # 3 arithmetic intervals:
        # 100 -> 110
        # 110 -> 120
        # 120 -> 130
        # Capital per interval = 100 USDT.
        self.grid = build_excel_grid_table(
            capital=300.0,
            ceiling=130.0,
            floor=100.0,
            gap=10.0,
            buy_fee=0.001,
            sell_fee=0.001,
        )

    def test_excel_template_checkpoint(self):
        """Known KZM Excel row must be reproduced exactly."""
        grid = build_excel_grid_table(
            capital=3000.0,
            ceiling=8987.0,
            floor=1987.0,
            gap=70.0,
            buy_fee=0.001,
            sell_fee=0.001,
        )

        first = grid.iloc[0]

        self.assertEqual(len(grid), 100)
        self.assertAlmostEqual(first["buy_price"], 8917.0, places=12)
        self.assertAlmostEqual(first["sell_price"], 8987.0, places=12)
        self.assertAlmostEqual(
            first["profit"],
            0.1750644398340242,
            places=12,
        )

    def test_grid_count_and_capital_per_level(self):
        """Number of grids must come from (ceiling-floor)/gap."""
        self.assertEqual(len(self.grid), 3)
        self.assertTrue(
            np.allclose(
                self.grid["capital_per_level"],
                100.0,
            )
        )

    def test_fee_formula_matches_excel_logic(self):
        """BUY fee is in base asset and SELL fee is in quote asset."""
        row = self.grid.loc[
            self.grid["buy_price"].eq(120.0)
        ].iloc[0]

        gross_base = 100.0 / 120.0
        expected_buy_fee_base = gross_base * 0.001
        expected_base = gross_base * 0.999
        expected_gross_sell = expected_base * 130.0
        expected_sell_fee = expected_gross_sell * 0.001
        expected_net_sell = expected_gross_sell * 0.999

        self.assertAlmostEqual(
            row["gross_base_amount"],
            gross_base,
            places=12,
        )
        self.assertAlmostEqual(
            row["buy_fee_base"],
            expected_buy_fee_base,
            places=12,
        )
        self.assertAlmostEqual(
            row["base_amount"],
            expected_base,
            places=12,
        )
        self.assertAlmostEqual(
            row["sell_fee_quote"],
            expected_sell_fee,
            places=12,
        )
        self.assertAlmostEqual(
            row["net_sell"],
            expected_net_sell,
            places=12,
        )

    def test_upward_move_does_not_trigger_buy(self):
        """A BUY must not be created merely because price moves upward."""
        data = make_candles(
            [
                (115.0, 125.0, 115.0, 122.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)

        self.assertTrue(result["trade_log"].empty)
        self.assertEqual(result["summary"]["open_positions"], 0)

    def test_downward_cross_triggers_buy(self):
        """Crossing 120 downward must create exactly one BUY at 120."""
        data = make_candles(
            [
                (125.0, 126.0, 115.0, 118.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)
        log = result["trade_log"]

        self.assertEqual(len(log), 1)
        self.assertEqual(log.iloc[0]["side"], "BUY")
        self.assertAlmostEqual(log.iloc[0]["price"], 120.0, places=12)
        self.assertEqual(result["summary"]["open_positions"], 1)

    def test_new_buy_cannot_sell_in_same_candle(self):
        """High above the target cannot close a position opened later in that candle."""
        data = make_candles(
            [
                (125.0, 135.0, 115.0, 120.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)

        self.assertEqual(
            result["trade_log"]["side"].tolist(),
            ["BUY"],
        )
        self.assertEqual(result["summary"]["completed_cycles"], 0)

    def test_buy_then_sell_on_later_candle(self):
        """120 BUY must close at its paired 130 SELL on a later candle."""
        data = make_candles(
            [
                (125.0, 126.0, 115.0, 118.0),
                (120.0, 131.0, 120.0, 130.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)
        log = result["trade_log"]

        self.assertEqual(log["side"].tolist(), ["BUY", "SELL"])
        self.assertAlmostEqual(log.iloc[0]["price"], 120.0, places=12)
        self.assertAlmostEqual(log.iloc[1]["price"], 130.0, places=12)
        self.assertEqual(result["summary"]["completed_cycles"], 1)

        reference = self.grid.loc[
            self.grid["buy_price"].eq(120.0)
        ].iloc[0]
        self.assertAlmostEqual(
            result["summary"]["realized_profit"],
            reference["profit"],
            places=12,
        )

    def test_multiple_grid_levels_fill_on_downward_cross(self):
        """A candle crossing 120, 110 and 100 may fill all three levels."""
        data = make_candles(
            [
                (125.0, 126.0, 95.0, 100.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)
        buys = result["trade_log"].loc[
            result["trade_log"]["side"].eq("BUY")
        ]

        self.assertEqual(len(buys), 3)
        self.assertEqual(
            set(buys["price"].tolist()),
            {100.0, 110.0, 120.0},
        )
        self.assertAlmostEqual(
            result["summary"]["final_cash"],
            0.0,
            places=10,
        )

    def test_holding_grid_cannot_buy_twice(self):
        """The same grid cannot open a second position before the first closes."""
        data = make_candles(
            [
                (125.0, 126.0, 115.0, 118.0),
                (125.0, 126.0, 115.0, 118.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)

        self.assertEqual(len(result["trade_log"]), 1)
        self.assertEqual(
            result["trade_log"].iloc[0]["side"],
            "BUY",
        )

    def test_sold_grid_cannot_rebuy_in_same_candle(self):
        """A SELL and downward cross in one candle must not recycle the same grid."""
        data = make_candles(
            [
                (125.0, 126.0, 115.0, 118.0),
                (125.0, 131.0, 115.0, 120.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)

        self.assertEqual(
            result["trade_log"]["side"].tolist(),
            ["BUY", "SELL"],
        )
        self.assertEqual(result["summary"]["open_positions"], 0)

    def test_insufficient_cash_prevents_overbuying(self):
        """Engine must stop filling when the available starting cash is exhausted."""
        data = make_candles(
            [
                (125.0, 126.0, 95.0, 100.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 100.0)
        buys = result["trade_log"].loc[
            result["trade_log"]["side"].eq("BUY")
        ]

        self.assertEqual(len(buys), 1)
        self.assertGreaterEqual(
            result["summary"]["final_cash"],
            -1e-9,
        )

    def test_cash_flow_reconciles(self):
        """Event cash flows must reproduce the final cash balance exactly."""
        data = make_candles(
            [
                (125.0, 126.0, 115.0, 118.0),
                (120.0, 131.0, 120.0, 130.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)
        log = result["trade_log"]

        self.assertTrue(
            np.allclose(
                log["cash_before"] + log["cash_flow"],
                log["cash_after"],
                atol=1e-10,
            )
        )
        self.assertAlmostEqual(
            300.0 + log["cash_flow"].sum(),
            result["summary"]["final_cash"],
            places=10,
        )

    def test_equity_identity(self):
        """Every candle equity must equal cash + BTC marked at candle close."""
        data = make_candles(
            [
                (125.0, 126.0, 115.0, 118.0),
                (120.0, 125.0, 117.0, 123.0),
            ]
        )

        result = run_grid_backtest(data, self.grid, 300.0)
        equity = result["equity_curve"]

        self.assertTrue(
            np.allclose(
                equity["cash"] + equity["btc"] * equity["close"],
                equity["equity"],
                atol=1e-10,
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
