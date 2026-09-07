import ast
import bisect
import heapq
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / 'Grid_trading.ipynb'


def load_notebook_function(function_name):
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))
    namespace = {'np': np, 'pd': pd, 'heapq': heapq, 'bisect': bisect}

    for cell in notebook['cells']:
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', '')
        if isinstance(source, list):
            source = ''.join(source)

        if f'def {function_name}(' not in source:
            continue

        tree = ast.parse(source)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                module = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(module)
                exec(
                    compile(module, str(NOTEBOOK_PATH), 'exec'),
                    namespace,
                )
                return namespace[function_name]

    raise RuntimeError(f'Function {function_name!r} not found.')


build_excel_grid_table = load_notebook_function('build_excel_grid_table')
run_grid_backtest = load_notebook_function('run_grid_backtest')
build_monthly_portfolio_pnl = load_notebook_function(
    'build_monthly_portfolio_pnl'
)


def make_candles(rows, start='2024-01-01'):
    rows = list(rows)
    times = pd.date_range(
        start,
        periods=len(rows),
        freq='min',
        tz='UTC',
    )
    return pd.DataFrame({
        'open_time': times,
        'open': [r[0] for r in rows],
        'high': [r[1] for r in rows],
        'low': [r[2] for r in rows],
        'close': [r[3] for r in rows],
    })


class TestGridTradingEngine(unittest.TestCase):
    def setUp(self):
        # 3 intervals: 100->110, 110->120, 120->130.
        # Capital = 300, therefore Cost/Grid = 100 USDT.
        self.grid = build_excel_grid_table(
            capital=300.0,
            ceiling=130.0,
            floor=100.0,
            gap=10.0,
            buy_fee=0.001,
            sell_fee=0.001,
        )

    def test_excel_template_checkpoint(self):
        grid = build_excel_grid_table(
            3000.0,
            8987.0,
            1987.0,
            70.0,
            0.001,
            0.001,
        )
        first = grid.iloc[0]

        self.assertEqual(len(grid), 100)
        self.assertAlmostEqual(first['buy_price'], 8917.0, places=12)
        self.assertAlmostEqual(first['sell_price'], 8987.0, places=12)
        self.assertAlmostEqual(
            first['profit'],
            0.1750644398340242,
            places=12,
        )

    def test_grid_count_and_capital_per_level(self):
        self.assertEqual(len(self.grid), 3)
        self.assertTrue(
            np.allclose(
                self.grid['capital_per_level'],
                100.0,
            )
        )

    def test_fee_formula_matches_excel_logic(self):
        row = self.grid.loc[
            self.grid['buy_price'].eq(120.0)
        ].iloc[0]

        gross_base = 100.0 / 120.0
        expected_buy_fee_base = gross_base * 0.001
        expected_base = gross_base * 0.999
        expected_gross_sell = expected_base * 130.0
        expected_sell_fee = expected_gross_sell * 0.001
        expected_net_sell = expected_gross_sell * 0.999

        self.assertAlmostEqual(
            row['gross_base_amount'],
            gross_base,
            places=12,
        )
        self.assertAlmostEqual(
            row['buy_fee_base'],
            expected_buy_fee_base,
            places=12,
        )
        self.assertAlmostEqual(
            row['base_amount'],
            expected_base,
            places=12,
        )
        self.assertAlmostEqual(
            row['sell_fee_quote'],
            expected_sell_fee,
            places=12,
        )
        self.assertAlmostEqual(
            row['net_sell'],
            expected_net_sell,
            places=12,
        )

    def test_upward_move_does_not_trigger_buy(self):
        result = run_grid_backtest(
            make_candles([
                (115, 125, 115, 122),
            ]),
            self.grid,
            300.0,
        )
        self.assertTrue(result['trade_log'].empty)
        self.assertEqual(
            result['summary']['open_positions'],
            0,
        )

    def test_downward_cross_triggers_buy(self):
        result = run_grid_backtest(
            make_candles([
                (125, 126, 115, 118),
            ]),
            self.grid,
            300.0,
        )
        log = result['trade_log']

        self.assertEqual(len(log), 1)
        self.assertEqual(log.iloc[0]['side'], 'BUY')
        self.assertAlmostEqual(
            log.iloc[0]['price'],
            120.0,
            places=12,
        )
        self.assertEqual(
            result['summary']['open_positions'],
            1,
        )

    def test_new_buy_cannot_sell_in_same_candle(self):
        result = run_grid_backtest(
            make_candles([
                (125, 135, 115, 120),
            ]),
            self.grid,
            300.0,
        )

        self.assertEqual(
            result['trade_log']['side'].tolist(),
            ['BUY'],
        )
        self.assertEqual(
            result['summary']['completed_cycles'],
            0,
        )

    def test_buy_then_sell_on_later_candle(self):
        data = make_candles([
            (125, 126, 115, 118),
            (120, 131, 120, 130),
        ])
        result = run_grid_backtest(
            data,
            self.grid,
            300.0,
        )
        log = result['trade_log']

        self.assertEqual(
            log['side'].tolist(),
            ['BUY', 'SELL'],
        )
        self.assertAlmostEqual(
            log.iloc[0]['price'],
            120.0,
            places=12,
        )
        self.assertAlmostEqual(
            log.iloc[1]['price'],
            130.0,
            places=12,
        )
        self.assertEqual(
            result['summary']['completed_cycles'],
            1,
        )

        self.assertAlmostEqual(
            result['summary']['realized_profit'],
            8.116775,
            places=12,
        )

    def test_multiple_grid_levels_fill_on_downward_cross(self):
        result = run_grid_backtest(
            make_candles([
                (125, 126, 95, 100),
            ]),
            self.grid,
            300.0,
        )
        buys = result['trade_log'].loc[
            result['trade_log']['side'].eq('BUY')
        ]

        self.assertEqual(len(buys), 3)
        self.assertEqual(
            set(buys['price'].tolist()),
            {100.0, 110.0, 120.0},
        )
        self.assertAlmostEqual(
            result['summary']['final_cash'],
            0.0,
            places=10,
        )

    def test_holding_grid_cannot_buy_twice(self):
        data = make_candles([
            (125, 126, 115, 118),
            (125, 126, 115, 118),
        ])
        result = run_grid_backtest(
            data,
            self.grid,
            300.0,
        )

        self.assertEqual(
            len(result['trade_log']),
            1,
        )
        self.assertEqual(
            result['trade_log'].iloc[0]['side'],
            'BUY',
        )

    def test_sold_grid_cannot_rebuy_in_same_candle(self):
        data = make_candles([
            (125, 126, 115, 118),
            (125, 131, 115, 120),
        ])
        result = run_grid_backtest(
            data,
            self.grid,
            300.0,
        )

        self.assertEqual(
            result['trade_log']['side'].tolist(),
            ['BUY', 'SELL'],
        )
        self.assertEqual(
            result['summary']['open_positions'],
            0,
        )

    def test_insufficient_cash_prevents_overbuying(self):
        result = run_grid_backtest(
            make_candles([
                (125, 126, 95, 100),
            ]),
            self.grid,
            100.0,
        )
        buys = result['trade_log'].loc[
            result['trade_log']['side'].eq('BUY')
        ]

        self.assertEqual(len(buys), 1)
        self.assertGreaterEqual(
            result['summary']['final_cash'],
            -1e-9,
        )

    def test_cash_movement_reconciles(self):
        data = make_candles([
            (125, 126, 115, 118),
            (120, 131, 120, 130),
        ])
        result = run_grid_backtest(
            data,
            self.grid,
            300.0,
        )
        log = result['trade_log']

        self.assertTrue(
            np.allclose(
                log['cash_before']
                + log['cash_movement'],
                log['cash_after'],
                atol=1e-10,
            )
        )
        self.assertAlmostEqual(
            300.0
            + log['cash_movement'].sum(),
            result['summary']['final_cash'],
            places=10,
        )

    def test_grid_cashflow_matches_independent_expected_profit(self):
        data = make_candles([
            (125, 126, 115, 118),
            (120, 131, 120, 130),
        ])
        result = run_grid_backtest(
            data,
            self.grid,
            300.0,
        )
        log = result['trade_log']

        self.assertAlmostEqual(
            log.iloc[0]['grid_cashflow'],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            log.iloc[1]['grid_cashflow'],
            8.116775,
            places=12,
        )
        self.assertAlmostEqual(
            log['grid_cashflow'].sum(),
            result['summary']['realized_profit'],
            places=12,
        )

        completed = result[
            'completed_trades'
        ].iloc[0]
        self.assertAlmostEqual(
            completed['actual_earn']
            - completed['cost'],
            completed['grid_cashflow'],
            places=12,
        )

    def test_equity_identity(self):
        data = make_candles([
            (125, 126, 115, 118),
            (120, 125, 117, 123),
        ])
        result = run_grid_backtest(
            data,
            self.grid,
            300.0,
        )
        equity = result['equity_curve']

        self.assertTrue(
            np.allclose(
                equity['cash']
                + equity['btc']
                * equity['close'],
                equity['equity'],
                atol=1e-10,
            )
        )

    def test_monthly_portfolio_pnl_known_values(self):
        equity_curve = pd.DataFrame({
            'open_time': pd.to_datetime([
                '2024-01-15T00:00:00Z',
                '2024-01-31T23:59:00Z',
                '2024-02-28T23:59:00Z',
            ]),
            'equity': [100.0, 110.0, 105.0],
        })

        monthly = build_monthly_portfolio_pnl(
            equity_curve=equity_curve,
            initial_capital=100.0,
            start_date='2024-01-01',
            end_date='2024-03-01',
        )

        self.assertEqual(
            monthly['month'].tolist(),
            ['2024-01', '2024-02'],
        )
        self.assertTrue(
            np.allclose(
                monthly['portfolio_net_pnl'],
                [10.0, -5.0],
            )
        )
        self.assertTrue(
            np.allclose(
                monthly['cumulative_portfolio_pnl'],
                [10.0, 5.0],
            )
        )

    def test_monthly_portfolio_pnl_reconciles_final_equity(self):
        equity_curve = pd.DataFrame({
            'open_time': pd.to_datetime([
                '2024-01-31T23:59:00Z',
                '2024-02-29T23:59:00Z',
                '2024-03-31T23:59:00Z',
            ]),
            'equity': [310.0, 295.0, 320.0],
        })

        monthly = build_monthly_portfolio_pnl(
            equity_curve=equity_curve,
            initial_capital=300.0,
            start_date='2024-01-01',
            end_date='2024-04-01',
        )

        self.assertAlmostEqual(
            monthly['portfolio_net_pnl'].sum(),
            20.0,
            places=12,
        )
        self.assertAlmostEqual(
            monthly['cumulative_portfolio_pnl'].iloc[-1],
            20.0,
            places=12,
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
