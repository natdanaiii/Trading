"""
V1 Dynamic Re-centering Grid + Risk
- User sets: Initial Capital, Initial Floor, Initial Ceiling, Gap, Fees,
  Recenter Trigger, and Risk Limits.
- Initial Reference = midpoint of Initial Floor/Ceiling.
- Old positions survive recenter and keep original SELL targets.
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

INITIAL_FLOOR = 27000.0
INITIAL_CEILING = 57000.0
GRID_GAP = 1000.0

BUY_FEE = 0.001
SELL_FEE = 0.001

RECENTER_TRIGGER_GRIDS = 5

MIN_CASH_RESERVE = 750.0
MAX_OPEN_POSITIONS = 18
MAX_DEPLOYED_CAPITAL = 1800.0
MAX_ENTRY_BTC_EXPOSURE = 1800.0
MAX_DRAWDOWN_STOP = 0.15

LIVE_EXECUTION_ENABLED = False


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


def derive_initial_grid():
    if INITIAL_CAPITAL <= 0:
        raise ValueError("INITIAL_CAPITAL must be > 0.")
    if INITIAL_FLOOR <= 0:
        raise ValueError("INITIAL_FLOOR must be > 0.")
    if INITIAL_CEILING <= INITIAL_FLOOR:
        raise ValueError("INITIAL_CEILING must be > INITIAL_FLOOR.")
    if GRID_GAP <= 0:
        raise ValueError("GRID_GAP must be > 0.")

    raw_count = (INITIAL_CEILING - INITIAL_FLOOR) / GRID_GAP
    if not np.isclose(raw_count, round(raw_count)):
        raise ValueError(
            "(INITIAL_CEILING - INITIAL_FLOOR) must be exactly divisible by GRID_GAP."
        )

    number_of_grids = int(round(raw_count))
    initial_reference = (INITIAL_FLOOR + INITIAL_CEILING) / 2.0

    if not np.isclose(initial_reference / GRID_GAP, round(initial_reference / GRID_GAP)):
        raise ValueError("Initial Reference midpoint must align with GRID_GAP.")

    capital_per_grid = INITIAL_CAPITAL / number_of_grids

    return number_of_grids, float(initial_reference), float(capital_per_grid)


NUMBER_OF_GRIDS, INITIAL_REFERENCE, CAPITAL_PER_GRID = derive_initial_grid()


def round_to_gap(price, gap):
    if gap <= 0:
        raise ValueError("gap must be > 0.")
    return float(np.floor(float(price) / gap + 0.5) * gap)


def build_regime(reference_price, gap, number_of_grids, capital, regime_id):
    lower_grids = number_of_grids // 2
    upper_grids = number_of_grids - lower_grids

    floor = reference_price - lower_grids * gap
    ceiling = reference_price + upper_grids * gap

    if floor <= 0:
        raise ValueError("Dynamic floor must stay > 0.")

    buy_prices = np.arange(floor, ceiling, gap, dtype=float)
    if len(buy_prices) != number_of_grids:
        raise AssertionError("Dynamic regime grid count mismatch.")

    return {
        "regime_id": int(regime_id),
        "reference_price": float(reference_price),
        "floor": float(floor),
        "ceiling": float(ceiling),
        "buy_prices": buy_prices,
        "capital_per_grid": float(capital / number_of_grids),
    }


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


def run_dynamic_grid_backtest(
    df_price,
    initial_capital,
    initial_reference,
    gap,
    number_of_grids,
    recenter_trigger_grids,
    buy_fee,
    sell_fee,
    min_cash_reserve,
    max_open_positions,
    max_deployed_capital,
    max_entry_btc_exposure,
    max_drawdown_stop,
):
    data = df_price.sort_values("open_time").reset_index(drop=True)
    if data.empty:
        raise ValueError("df_price is empty.")

    regime_id = 0
    regime = build_regime(
        initial_reference,
        gap,
        number_of_grids,
        initial_capital,
        regime_id,
    )

    regime_history = [
        {
            "regime_id": 0,
            "effective_time": data.iloc[0]["open_time"],
            "reference_price": regime["reference_price"],
            "floor": regime["floor"],
            "ceiling": regime["ceiling"],
            "reason": "INITIAL",
        }
    ]
    recenter_events = []
    risk_halt_events = []

    cash = float(initial_capital)
    btc = 0.0
    deployed_capital = 0.0
    realized_profit = 0.0
    total_buy_fee_usdt = 0.0
    total_sell_fee_usdt = 0.0
    completed_cycles = 0

    positions = {}
    open_by_buy_price = {}
    sell_heap = []

    events = []
    completed = []

    blocked = {
        "risk_halt": 0,
        "cash_reserve": 0,
        "max_open_positions": 0,
        "max_deployed_capital": 0,
        "max_entry_btc_exposure": 0,
    }

    n = len(data)
    equity_values = np.empty(n)
    cash_values = np.empty(n)
    btc_values = np.empty(n)
    deployed_values = np.empty(n)
    open_position_values = np.empty(n, dtype=int)
    btc_market_value_values = np.empty(n)
    reference_values = np.empty(n)
    regime_values = np.empty(n, dtype=int)
    risk_halt_values = np.empty(n, dtype=bool)

    previous_close = None
    risk_halt = False
    peak_equity = float(initial_capital)

    event_id = 0
    position_id = 0
    tolerance = 1e-9

    for i, row in enumerate(data.itertuples(index=False)):
        timestamp = row.open_time
        open_price = float(row.open)
        high_price = float(row.high)
        low_price = float(row.low)
        close_price = float(row.close)

        cash_at_candle_start = cash
        sold_this_candle = set()

        while sell_heap and sell_heap[0][0] <= high_price + tolerance:
            _, pid = heapq.heappop(sell_heap)
            position = positions.get(pid)

            if position is None or not position["is_open"]:
                continue

            cash_before = cash
            btc_before = btc

            position["is_open"] = False
            cash += position["net_sell"]
            btc -= position["base_amount"]
            deployed_capital -= position["cost"]

            if abs(btc) < 1e-12:
                btc = 0.0
            if abs(deployed_capital) < 1e-10:
                deployed_capital = 0.0

            realized_profit += position["profit"]
            total_sell_fee_usdt += position["sell_fee_quote"]
            completed_cycles += 1

            sold_this_candle.add(position["buy_price"])
            open_by_buy_price.pop(position["buy_price"], None)
            event_id += 1

            completed.append(
                {
                    "position_id": pid,
                    "regime_id": position["regime_id"],
                    "buy_time": position["buy_time"],
                    "sell_time": timestamp,
                    "buy_price": position["buy_price"],
                    "sell_price": position["sell_price"],
                    "cost": position["cost"],
                    "profit": position["profit"],
                }
            )

            events.append(
                {
                    "event_id": event_id,
                    "time": timestamp,
                    "side": "SELL",
                    "position_id": pid,
                    "regime_id": position["regime_id"],
                    "price": position["sell_price"],
                    "cash_movement": position["net_sell"],
                    "grid_cashflow": position["profit"],
                    "cash_before": cash_before,
                    "cash_after": cash,
                    "btc_before": btc_before,
                    "btc_after": btc,
                    "deployed_capital_after": deployed_capital,
                }
            )

        buy_budget = cash_at_candle_start
        downward_start = (
            open_price if previous_close is None else max(previous_close, open_price)
        )

        active_buy_prices = regime["buy_prices"]
        active_buy_price_list = active_buy_prices.tolist()

        if low_price < downward_start:
            first_index = bisect.bisect_left(active_buy_price_list, low_price)
            stop_index = bisect.bisect_left(active_buy_price_list, downward_start)

            for k in range(stop_index - 1, first_index - 1, -1):
                buy_price = float(active_buy_prices[k])

                if buy_price in open_by_buy_price or buy_price in sold_this_candle:
                    continue

                if risk_halt:
                    blocked["risk_halt"] += 1
                    break

                cost = regime["capital_per_grid"]

                if buy_budget + tolerance < cost:
                    break
                if buy_budget - cost < min_cash_reserve - tolerance:
                    blocked["cash_reserve"] += 1
                    break
                if (
                    max_open_positions is not None
                    and len(open_by_buy_price) >= max_open_positions
                ):
                    blocked["max_open_positions"] += 1
                    break
                if (
                    max_deployed_capital is not None
                    and deployed_capital + cost > max_deployed_capital + tolerance
                ):
                    blocked["max_deployed_capital"] += 1
                    break

                sell_price = buy_price + gap
                gross_base = cost / buy_price
                buy_fee_base = gross_base * buy_fee
                base_amount = gross_base - buy_fee_base

                projected_btc = btc + base_amount
                projected_entry_exposure = projected_btc * buy_price

                if (
                    max_entry_btc_exposure is not None
                    and projected_entry_exposure > max_entry_btc_exposure + tolerance
                ):
                    blocked["max_entry_btc_exposure"] += 1
                    break

                gross_sell = base_amount * sell_price
                sell_fee_quote = gross_sell * sell_fee
                net_sell = gross_sell - sell_fee_quote
                cycle_profit = net_sell - cost

                cash_before = cash
                btc_before = btc

                buy_budget -= cost
                cash -= cost
                btc += base_amount
                deployed_capital += cost
                total_buy_fee_usdt += buy_fee_base * buy_price

                position_id += 1
                positions[position_id] = {
                    "position_id": position_id,
                    "regime_id": regime["regime_id"],
                    "reference_price": regime["reference_price"],
                    "buy_time": timestamp,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "cost": cost,
                    "base_amount": base_amount,
                    "sell_fee_quote": sell_fee_quote,
                    "net_sell": net_sell,
                    "profit": cycle_profit,
                    "is_open": True,
                }

                open_by_buy_price[buy_price] = position_id
                heapq.heappush(sell_heap, (sell_price, position_id))

                event_id += 1
                events.append(
                    {
                        "event_id": event_id,
                        "time": timestamp,
                        "side": "BUY",
                        "position_id": position_id,
                        "regime_id": regime["regime_id"],
                        "price": buy_price,
                        "cash_movement": -cost,
                        "grid_cashflow": 0.0,
                        "cash_before": cash_before,
                        "cash_after": cash,
                        "btc_before": btc_before,
                        "btc_after": btc,
                        "deployed_capital_after": deployed_capital,
                        "entry_exposure_after": projected_entry_exposure,
                    }
                )

        equity = cash + btc * close_price
        peak_equity = max(peak_equity, equity)
        current_drawdown = equity / peak_equity - 1.0

        if (
            not risk_halt
            and max_drawdown_stop is not None
            and current_drawdown <= -max_drawdown_stop
        ):
            risk_halt = True
            risk_halt_events.append(
                {
                    "decision_time": timestamp,
                    "effective_time": (
                        data.iloc[i + 1]["open_time"] if i + 1 < n else pd.NaT
                    ),
                    "drawdown": float(current_drawdown),
                    "equity": float(equity),
                    "peak_equity": float(peak_equity),
                }
            )

        equity_values[i] = equity
        cash_values[i] = cash
        btc_values[i] = btc
        deployed_values[i] = deployed_capital
        open_position_values[i] = len(open_by_buy_price)
        btc_market_value_values[i] = btc * close_price
        reference_values[i] = regime["reference_price"]
        regime_values[i] = regime["regime_id"]
        risk_halt_values[i] = risk_halt

        trigger_distance = recenter_trigger_grids * gap

        if (
            close_price >= regime["reference_price"] + trigger_distance
            or close_price <= regime["reference_price"] - trigger_distance
        ):
            new_reference = round_to_gap(close_price, gap)

            if new_reference != regime["reference_price"]:
                old_regime = regime
                regime_id += 1

                regime = build_regime(
                    new_reference,
                    gap,
                    number_of_grids,
                    initial_capital,
                    regime_id,
                )

                direction = (
                    "UP"
                    if new_reference > old_regime["reference_price"]
                    else "DOWN"
                )

                recenter_events.append(
                    {
                        "decision_time": timestamp,
                        "direction": direction,
                        "close": close_price,
                        "old_reference": old_regime["reference_price"],
                        "new_reference": new_reference,
                        "old_floor": old_regime["floor"],
                        "old_ceiling": old_regime["ceiling"],
                        "new_floor": regime["floor"],
                        "new_ceiling": regime["ceiling"],
                    }
                )

                regime_history.append(
                    {
                        "regime_id": regime_id,
                        "effective_time": (
                            data.iloc[i + 1]["open_time"] if i + 1 < n else pd.NaT
                        ),
                        "reference_price": regime["reference_price"],
                        "floor": regime["floor"],
                        "ceiling": regime["ceiling"],
                        "reason": f"RECENTER_{direction}",
                    }
                )

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
            "btc_market_value": btc_market_value_values,
            "deployed_capital": deployed_values,
            "open_positions": open_position_values,
            "equity": equity_values,
            "reference_price": reference_values,
            "regime_id": regime_values,
            "risk_halt": risk_halt_values,
            "drawdown": drawdown,
        }
    )

    trade_log = pd.DataFrame(events)
    open_positions = [p for p in positions.values() if p["is_open"]]

    max_entry_exposure_observed = 0.0
    if len(trade_log) and "entry_exposure_after" in trade_log.columns:
        buy_events = trade_log.loc[trade_log["side"].eq("BUY")]
        if len(buy_events):
            max_entry_exposure_observed = float(
                buy_events["entry_exposure_after"].max()
            )

    summary = {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "net_return": net_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "completed_cycles": completed_cycles,
        "open_positions": len(open_positions),
        "final_cash": cash,
        "final_btc": btc,
        "final_deployed_capital": deployed_capital,
        "realized_profit": realized_profit,
        "unrealized_pnl": final_equity - initial_capital - realized_profit,
        "total_fee_usdt_equiv": total_buy_fee_usdt + total_sell_fee_usdt,
        "recenter_count": len(recenter_events),
        "risk_halt_triggered": risk_halt,
        "risk_halt_count": len(risk_halt_events),
        "min_cash_observed": float(cash_values.min()),
        "max_open_positions_observed": int(open_position_values.max()),
        "max_deployed_capital_observed": float(deployed_values.max()),
        "max_btc_market_value_observed": float(btc_market_value_values.max()),
        "max_entry_btc_exposure_observed": max_entry_exposure_observed,
        "blocked_buy_counts": blocked,
    }

    return {
        "summary": summary,
        "trade_log": trade_log,
        "completed_trades": pd.DataFrame(completed),
        "equity_curve": equity_curve,
        "open_positions": pd.DataFrame(open_positions),
        "recenter_log": pd.DataFrame(recenter_events),
        "regime_history": pd.DataFrame(regime_history),
        "risk_halt_log": pd.DataFrame(risk_halt_events),
    }


def audit_v1(result):
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

    recenter_log = result["recenter_log"]
    recenter_ok = True
    if len(recenter_log):
        recenter_ok = (
            (recenter_log["close"] - recenter_log["old_reference"]).abs()
            + 1e-9
            >= RECENTER_TRIGGER_GRIDS * GRID_GAP
        ).all()

    post_halt_buys = 0
    halt_log = result["risk_halt_log"]
    if len(halt_log):
        effective_time = halt_log.iloc[0]["effective_time"]
        if not pd.isna(effective_time):
            post_halt_buys = int(
                (
                    log["side"].eq("BUY")
                    & (pd.to_datetime(log["time"], utc=True) >= pd.Timestamp(effective_time))
                ).sum()
            )

    return {
        "cash_reconciliation": cash_error <= 1e-8,
        "realized_profit_reconciliation": realized_error <= 1e-8,
        "equity_identity": equity_error <= 1e-8,
        "cash_never_negative": float(equity["cash"].min()) >= -1e-8,
        "recenter_threshold": bool(recenter_ok),
        "cash_reserve": summary["min_cash_observed"] >= MIN_CASH_RESERVE - 1e-8,
        "max_open_positions": summary["max_open_positions_observed"] <= MAX_OPEN_POSITIONS,
        "max_deployed_capital": (
            summary["max_deployed_capital_observed"] <= MAX_DEPLOYED_CAPITAL + 1e-8
        ),
        "max_entry_btc_exposure": (
            summary["max_entry_btc_exposure_observed"] <= MAX_ENTRY_BTC_EXPOSURE + 1e-8
        ),
        "no_buy_after_risk_halt": post_halt_buys == 0,
    }


if __name__ == "__main__":
    df_1m = load_market_data(SYMBOL, TIMEFRAME, DATA_DIR)

    result = run_dynamic_grid_backtest(
        df_price=df_1m,
        initial_capital=INITIAL_CAPITAL,
        initial_reference=INITIAL_REFERENCE,
        gap=GRID_GAP,
        number_of_grids=NUMBER_OF_GRIDS,
        recenter_trigger_grids=RECENTER_TRIGGER_GRIDS,
        buy_fee=BUY_FEE,
        sell_fee=SELL_FEE,
        min_cash_reserve=MIN_CASH_RESERVE,
        max_open_positions=MAX_OPEN_POSITIONS,
        max_deployed_capital=MAX_DEPLOYED_CAPITAL,
        max_entry_btc_exposure=MAX_ENTRY_BTC_EXPOSURE,
        max_drawdown_stop=MAX_DRAWDOWN_STOP,
    )

    summary = result["summary"]

    print("===== V1 CONFIGURATION =====")
    print(f"Capital           : {INITIAL_CAPITAL:,.2f} USDT")
    print(f"Initial Floor     : {INITIAL_FLOOR:,.2f}")
    print(f"Initial Ceiling   : {INITIAL_CEILING:,.2f}")
    print(f"Initial Reference : {INITIAL_REFERENCE:,.2f}")
    print(f"Gap               : {GRID_GAP:,.2f}")
    print(f"Number of Grids   : {NUMBER_OF_GRIDS}")
    print(f"Capital / Grid    : {CAPITAL_PER_GRID:,.2f} USDT")
    print(f"Recenter Trigger  : ±{RECENTER_TRIGGER_GRIDS * GRID_GAP:,.0f} USDT")

    print("\n===== V1 RESULT =====")
    print(f"Final Equity      : {summary['final_equity']:,.2f} USDT")
    print(f"Net Return        : {summary['net_return']:.2%}")
    print(f"Max Drawdown      : {summary['max_drawdown']:.2%}")
    print(f"Calmar Ratio      : {summary['calmar_ratio']:.3f}")
    print(f"Cycles            : {summary['completed_cycles']:,}")
    print(f"Open Positions    : {summary['open_positions']:,}")
    print(f"Recenter Count    : {summary['recenter_count']:,}")
    print(f"Risk Halt         : {summary['risk_halt_triggered']}")
    print(f"Blocked BUYs      : {summary['blocked_buy_counts']}")

    checks = audit_v1(result)
    print("\n===== V1 AUDIT =====")
    for name, passed in checks.items():
        print(f"{name:32s}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise AssertionError("V1 AUDIT FAILED")
