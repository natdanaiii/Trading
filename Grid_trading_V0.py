"""
V0 Fixed Grid Backtest
- User sets: Initial Capital, Floor, Ceiling, Gap, Fees.
- Floor/Ceiling are NOT derived from future historical High/Low.
"""

import os
import heapq
import bisect

import numpy as np
import pandas as pd

# ============================================================
# USER CONFIGURATION
# ============================================================
DATA_DIR = "/content/drive/MyDrive/03.Trading/00.Live Trading"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
START_DATE = "2024-01-01"
END_DATE = "2026-01-01"

INITIAL_CAPITAL = 3000.0
GRID_FLOOR = 38000.0
GRID_CEILING = 127000.0
GRID_GAP = 1000.0

BUY_FEE = 0.001
SELL_FEE = 0.001


def load_market_data(symbol, timeframe, data_dir):
    path = os.path.join(data_dir, f"{symbol}-{timeframe}-combined.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    numeric = ["open", "high", "low", "close", "volume"]
    df[numeric] = df[numeric].astype(float)

    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts = pd.Timestamp(END_DATE, tz="UTC")

    return (
        df.drop_duplicates("open_time")
        .sort_values("open_time")
        .loc[lambda x: (x["open_time"] >= start_ts) & (x["open_time"] < end_ts)]
        .reset_index(drop=True)
    )


def build_fixed_grid_table(capital, floor, ceiling, gap, buy_fee, sell_fee):
    if capital <= 0:
        raise ValueError("capital must be > 0.")
    if floor <= 0:
        raise ValueError("floor must be > 0.")
    if ceiling <= floor:
        raise ValueError("ceiling must be > floor.")
    if gap <= 0:
        raise ValueError("gap must be > 0.")
    if not (0 <= buy_fee < 1 and 0 <= sell_fee < 1):
        raise ValueError("fees must be in [0, 1).")

    raw_count = (ceiling - floor) / gap
    if not np.isclose(raw_count, round(raw_count)):
        raise ValueError("(ceiling - floor) must be exactly divisible by gap.")

    number_of_grids = int(round(raw_count))
    capital_per_grid = capital / number_of_grids

    buy_prices = ceiling - gap * np.arange(1, number_of_grids + 1)
    sell_prices = buy_prices + gap

    gross_base = capital_per_grid / buy_prices
    buy_fee_base = gross_base * buy_fee
    base_amount = gross_base - buy_fee_base

    gross_sell = base_amount * sell_prices
    sell_fee_quote = gross_sell * sell_fee
    net_sell = gross_sell - sell_fee_quote
    profit = net_sell - capital_per_grid

    return pd.DataFrame(
        {
            "level": np.arange(1, number_of_grids + 1),
            "buy_price": buy_prices,
            "sell_price": sell_prices,
            "capital_per_grid": capital_per_grid,
            "base_amount": base_amount,
            "buy_fee_base": buy_fee_base,
            "sell_fee_quote": sell_fee_quote,
            "net_sell": net_sell,
            "profit": profit,
        }
    )


def performance_stats(data, equity, initial_capital):
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0

    final_equity = float(equity[-1])
    max_drawdown = float(drawdown.min())
    net_return = final_equity / initial_capital - 1.0

    elapsed_days = (
        data["open_time"].iloc[-1] - data["open_time"].iloc[0]
    ).total_seconds() / 86400.0

    annualized_return = np.nan
    if elapsed_days > 0 and final_equity > 0:
        growth = np.log(final_equity / initial_capital) * (365.25 / elapsed_days)
        if growth < 700:
            annualized_return = float(np.expm1(growth))

    calmar = np.nan
    if max_drawdown < 0 and np.isfinite(annualized_return):
        calmar = float(annualized_return / abs(max_drawdown))

    return final_equity, net_return, annualized_return, max_drawdown, calmar, drawdown


def run_fixed_grid_backtest(df_price, grid, initial_capital):
    data = df_price.sort_values("open_time").reset_index(drop=True)
    grid = grid.sort_values("buy_price").reset_index(drop=True).copy()

    if data.empty:
        raise ValueError("df_price is empty.")
    if grid.empty:
        raise ValueError("grid is empty.")

    buy_prices = grid["buy_price"].to_numpy(float)
    sell_prices = grid["sell_price"].to_numpy(float)
    costs = grid["capital_per_grid"].to_numpy(float)
    base_amounts = grid["base_amount"].to_numpy(float)
    buy_fee_base = grid["buy_fee_base"].to_numpy(float)
    sell_fee_quote = grid["sell_fee_quote"].to_numpy(float)
    net_sell = grid["net_sell"].to_numpy(float)
    cycle_profit = grid["profit"].to_numpy(float)

    holding = np.zeros(len(grid), dtype=bool)
    buy_time = [None] * len(grid)
    sell_heap = []

    cash = float(initial_capital)
    btc = 0.0
    realized_profit = 0.0
    total_buy_fee_btc = 0.0
    total_buy_fee_usdt = 0.0
    total_sell_fee_usdt = 0.0
    completed_cycles = 0

    events = []
    completed = []

    equity_values = np.empty(len(data))
    cash_values = np.empty(len(data))
    btc_values = np.empty(len(data))

    buy_price_list = buy_prices.tolist()
    previous_close = None
    event_id = 0

    for i, row in enumerate(data.itertuples(index=False)):
        timestamp = row.open_time
        open_price = float(row.open)
        high_price = float(row.high)
        low_price = float(row.low)
        close_price = float(row.close)

        cash_at_candle_start = cash
        sold_this_candle = set()

        while sell_heap and sell_heap[0][0] <= high_price:
            _, k = heapq.heappop(sell_heap)
            if not holding[k]:
                continue

            cash_before = cash
            btc_before = btc

            holding[k] = False
            cash += net_sell[k]
            btc -= base_amounts[k]
            if abs(btc) < 1e-12:
                btc = 0.0

            realized_profit += cycle_profit[k]
            total_sell_fee_usdt += sell_fee_quote[k]
            completed_cycles += 1
            sold_this_candle.add(k)
            event_id += 1

            completed.append(
                {
                    "buy_time": buy_time[k],
                    "sell_time": timestamp,
                    "buy_price": buy_prices[k],
                    "sell_price": sell_prices[k],
                    "cost": costs[k],
                    "profit": cycle_profit[k],
                }
            )

            events.append(
                {
                    "event_id": event_id,
                    "time": timestamp,
                    "side": "SELL",
                    "price": sell_prices[k],
                    "cash_movement": net_sell[k],
                    "grid_cashflow": cycle_profit[k],
                    "cash_before": cash_before,
                    "cash_after": cash,
                    "btc_before": btc_before,
                    "btc_after": btc,
                }
            )
            buy_time[k] = None

        buy_budget = cash_at_candle_start
        downward_start = (
            open_price if previous_close is None else max(previous_close, open_price)
        )

        if low_price < downward_start:
            first_index = bisect.bisect_left(buy_price_list, low_price)
            stop_index = bisect.bisect_left(buy_price_list, downward_start)

            for k in range(stop_index - 1, first_index - 1, -1):
                if holding[k] or k in sold_this_candle:
                    continue
                if buy_budget + 1e-12 < costs[k]:
                    break

                cash_before = cash
                btc_before = btc

                holding[k] = True
                buy_time[k] = timestamp
                buy_budget -= costs[k]
                cash -= costs[k]
                btc += base_amounts[k]

                total_buy_fee_btc += buy_fee_base[k]
                total_buy_fee_usdt += buy_fee_base[k] * buy_prices[k]

                heapq.heappush(sell_heap, (sell_prices[k], k))
                event_id += 1

                events.append(
                    {
                        "event_id": event_id,
                        "time": timestamp,
                        "side": "BUY",
                        "price": buy_prices[k],
                        "cash_movement": -costs[k],
                        "grid_cashflow": 0.0,
                        "cash_before": cash_before,
                        "cash_after": cash,
                        "btc_before": btc_before,
                        "btc_after": btc,
                    }
                )

        equity_values[i] = cash + btc * close_price
        cash_values[i] = cash
        btc_values[i] = btc
        previous_close = close_price

    final_equity, net_return, annualized_return, max_drawdown, calmar, drawdown = (
        performance_stats(data, equity_values, initial_capital)
    )

    equity_curve = pd.DataFrame(
        {
            "open_time": data["open_time"],
            "close": data["close"],
            "cash": cash_values,
            "btc": btc_values,
            "equity": equity_values,
            "drawdown": drawdown,
        }
    )

    trade_log = pd.DataFrame(events)

    summary = {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "net_return": net_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "completed_cycles": completed_cycles,
        "open_positions": int(holding.sum()),
        "final_cash": cash,
        "final_btc": btc,
        "realized_profit": realized_profit,
        "unrealized_pnl": final_equity - initial_capital - realized_profit,
        "total_fee_usdt_equiv": total_buy_fee_usdt + total_sell_fee_usdt,
    }

    return {
        "summary": summary,
        "trade_log": trade_log,
        "completed_trades": pd.DataFrame(completed),
        "equity_curve": equity_curve,
        "grid_state": grid.assign(holding=holding, buy_time=buy_time),
    }


def audit_v0(result):
    summary = result["summary"]
    log = result["trade_log"]
    equity = result["equity_curve"]

    cash_movement = log["cash_movement"].sum() if len(log) else 0.0
    grid_cashflow = log["grid_cashflow"].sum() if len(log) else 0.0

    cash_error = abs(INITIAL_CAPITAL + cash_movement - summary["final_cash"])
    realized_error = abs(grid_cashflow - summary["realized_profit"])
    equity_error = float(
        np.max(np.abs(equity["cash"] + equity["btc"] * equity["close"] - equity["equity"]))
    )

    return {
        "cash_reconciliation": cash_error <= 1e-8,
        "realized_profit_reconciliation": realized_error <= 1e-8,
        "equity_identity": equity_error <= 1e-8,
        "cash_never_negative": float(equity["cash"].min()) >= -1e-8,
    }


if __name__ == "__main__":
    df_1m = load_market_data(SYMBOL, TIMEFRAME, DATA_DIR)

    grid = build_fixed_grid_table(
        INITIAL_CAPITAL,
        GRID_FLOOR,
        GRID_CEILING,
        GRID_GAP,
        BUY_FEE,
        SELL_FEE,
    )

    result = run_fixed_grid_backtest(df_1m, grid, INITIAL_CAPITAL)
    summary = result["summary"]

    print("===== V0 CONFIGURATION =====")
    print(f"Capital         : {INITIAL_CAPITAL:,.2f} USDT")
    print(f"Floor           : {GRID_FLOOR:,.2f}")
    print(f"Ceiling         : {GRID_CEILING:,.2f}")
    print(f"Gap             : {GRID_GAP:,.2f}")
    print(f"Number of Grids : {len(grid)}")
    print(f"Capital / Grid  : {grid['capital_per_grid'].iloc[0]:,.6f} USDT")

    print("\n===== V0 RESULT =====")
    print(f"Final Equity    : {summary['final_equity']:,.2f} USDT")
    print(f"Net Return      : {summary['net_return']:.2%}")
    print(f"Max Drawdown    : {summary['max_drawdown']:.2%}")
    print(f"Calmar Ratio    : {summary['calmar_ratio']:.3f}")
    print(f"Cycles          : {summary['completed_cycles']:,}")
    print(f"Open Positions  : {summary['open_positions']:,}")

    checks = audit_v0(result)
    print("\n===== V0 AUDIT =====")
    for name, passed in checks.items():
        print(f"{name:32s}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise AssertionError("V0 AUDIT FAILED")
