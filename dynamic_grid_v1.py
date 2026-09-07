"""V1 dynamic re-centering grid engine.

The V1 design intentionally keeps order sizing fixed within the run and changes
only the active grid regime through causal re-centering. Existing positions keep
their original sell targets after a re-center.
"""

import bisect
import heapq

import numpy as np
import pandas as pd


def round_to_gap(price, gap):
    """Round a positive price to the nearest arithmetic grid step."""
    if gap <= 0:
        raise ValueError("gap must be greater than 0.")
    return float(np.floor(float(price) / gap + 0.5) * gap)


def build_dynamic_regime(reference_price, gap, number_of_grids, capital, regime_id=0):
    """Build one centered arithmetic grid regime.

    `number_of_grids` is the number of BUY->SELL intervals. For an even number,
    half sit below and half above the reference. Capital per grid is fixed at
    `capital / number_of_grids` for the entire V1 run.
    """
    if capital <= 0:
        raise ValueError("capital must be greater than 0.")
    if gap <= 0:
        raise ValueError("gap must be greater than 0.")
    if number_of_grids < 2:
        raise ValueError("number_of_grids must be at least 2.")

    lower_grids = number_of_grids // 2
    upper_grids = number_of_grids - lower_grids

    floor = float(reference_price - lower_grids * gap)
    ceiling = float(reference_price + upper_grids * gap)

    if floor <= 0:
        raise ValueError(
            "Dynamic grid floor must stay above 0. Reduce number_of_grids or gap."
        )

    buy_prices = np.arange(floor, ceiling, gap, dtype=float)
    if len(buy_prices) != number_of_grids:
        raise AssertionError("Dynamic regime grid count mismatch.")

    return {
        "regime_id": int(regime_id),
        "reference_price": float(reference_price),
        "floor": floor,
        "ceiling": ceiling,
        "buy_prices": buy_prices,
        "capital_per_grid": float(capital / number_of_grids),
        "gap": float(gap),
    }


def run_dynamic_grid_backtest(
    df_price,
    initial_capital,
    gap,
    number_of_grids,
    recenter_trigger_grids,
    buy_fee=0.001,
    sell_fee=0.001,
):
    """Run V1 dynamic re-centering grid backtest on OHLC candles.

    Execution semantics intentionally match V0 where applicable:
    - BUY only on a downward crossing of a current-regime grid level.
    - Existing SELL targets fill when candle High reaches the target.
    - New BUYs cannot SELL in the same candle.
    - A level sold in a candle cannot rebuy in that same candle.
    - Same-candle SELL proceeds are not reused for BUYs.
    - BUY fee is deducted from BTC; SELL fee is deducted from USDT.

    Dynamic-grid rule:
    - Initial reference uses only the first candle Open.
    - A re-center decision uses the candle Close and is effective next candle.
    - Trigger distance = recenter_trigger_grids * gap.
    - Existing positions survive a re-center and retain their original SELL target.
    - Order size does not compound in V1.
    """
    required = {"open_time", "open", "high", "low", "close"}
    missing = required.difference(df_price.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df_price) == 0:
        raise ValueError("df_price is empty.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be greater than 0.")
    if gap <= 0:
        raise ValueError("gap must be greater than 0.")
    if number_of_grids < 2:
        raise ValueError("number_of_grids must be at least 2.")
    if recenter_trigger_grids < 1:
        raise ValueError("recenter_trigger_grids must be at least 1.")
    if not (0 <= buy_fee < 1 and 0 <= sell_fee < 1):
        raise ValueError("fees must be in [0, 1).")

    data = df_price.sort_values("open_time").reset_index(drop=True)

    initial_reference = round_to_gap(float(data.iloc[0]["open"]), gap)
    regime_id = 0
    regime = build_dynamic_regime(
        initial_reference,
        gap,
        number_of_grids,
        initial_capital,
        regime_id,
    )

    regime_history = [{
        "regime_id": 0,
        "effective_time": data.iloc[0]["open_time"],
        "reference_price": regime["reference_price"],
        "floor": regime["floor"],
        "ceiling": regime["ceiling"],
        "reason": "INITIAL",
    }]
    recenter_events = []

    cash = float(initial_capital)
    open_btc = 0.0
    realized_profit = 0.0
    total_buy_fee_btc = 0.0
    total_buy_fee_usdt_equiv = 0.0
    total_sell_fee_usdt = 0.0
    completed_cycles = 0

    event_id = 0
    position_id = 0
    positions = {}
    open_by_buy_price = {}
    sell_heap = []
    trade_events = []
    completed_trades = []

    n_rows = len(data)
    equity_values = np.empty(n_rows)
    cash_values = np.empty(n_rows)
    btc_values = np.empty(n_rows)
    reference_values = np.empty(n_rows)
    regime_values = np.empty(n_rows, dtype=int)

    prev_close = None
    tolerance = 1e-9

    for i, row in enumerate(data.itertuples(index=False)):
        timestamp = row.open_time
        open_price = float(row.open)
        high_price = float(row.high)
        low_price = float(row.low)
        close_price = float(row.close)

        cash_at_candle_start = cash
        sold_this_candle = set()

        # 1) Existing SELL orders, including positions from older regimes.
        while sell_heap and sell_heap[0][0] <= high_price + tolerance:
            _, pid = heapq.heappop(sell_heap)
            pos = positions.get(pid)

            if pos is None or not pos["is_open"]:
                continue

            cash_before = cash
            btc_before = open_btc

            pos["is_open"] = False
            cash += pos["net_sell"]
            open_btc -= pos["base_amount"]
            if abs(open_btc) < 1e-12:
                open_btc = 0.0

            realized_profit += pos["profit"]
            total_sell_fee_usdt += pos["sell_fee_quote"]
            completed_cycles += 1
            sold_this_candle.add(pos["buy_price"])
            open_by_buy_price.pop(pos["buy_price"], None)
            event_id += 1

            completed_trades.append({
                "position_id": pid,
                "regime_id": pos["regime_id"],
                "buy_time": pos["buy_time"],
                "sell_time": timestamp,
                "buy_price": pos["buy_price"],
                "sell_price": pos["sell_price"],
                "cost": pos["cost"],
                "quote_cost": pos["cost"],
                "base_amount": pos["base_amount"],
                "actual_earn": pos["net_sell"],
                "net_sell": pos["net_sell"],
                "grid_cashflow": pos["profit"],
                "profit": pos["profit"],
            })

            trade_events.append({
                "event_id": event_id,
                "time": timestamp,
                "side": "SELL",
                "position_id": pid,
                "regime_id": pos["regime_id"],
                "reference_price": pos["reference_price"],
                "price": pos["sell_price"],
                "base_amount": pos["base_amount"],
                "quote_amount": pos["net_sell"],
                "fee_base": 0.0,
                "fee_quote": pos["sell_fee_quote"],
                "realized_profit": pos["profit"],
                "cash_movement": pos["net_sell"],
                "grid_cashflow": pos["profit"],
                "cash_before": cash_before,
                "cash_after": cash,
                "btc_before": btc_before,
                "btc_after": open_btc,
            })

        # 2) Downward BUY crossings in the current regime.
        # Same-candle SELL proceeds are intentionally unavailable here.
        buy_budget = cash_at_candle_start
        down_start = (
            open_price if prev_close is None else max(prev_close, open_price)
        )

        buy_prices = regime["buy_prices"]
        buy_price_list = buy_prices.tolist()

        if low_price < down_start:
            first_idx = bisect.bisect_left(buy_price_list, low_price)
            stop_idx = bisect.bisect_left(buy_price_list, down_start)

            # Higher levels are crossed first on a downward move.
            for k in range(stop_idx - 1, first_idx - 1, -1):
                buy_price = float(buy_prices[k])

                if (
                    buy_price in open_by_buy_price
                    or buy_price in sold_this_candle
                ):
                    continue

                cost = regime["capital_per_grid"]
                if buy_budget + 1e-12 < cost:
                    break

                sell_price = buy_price + gap
                gross_base_amount = cost / buy_price
                buy_fee_base = gross_base_amount * buy_fee
                base_amount = gross_base_amount - buy_fee_base

                gross_sell = base_amount * sell_price
                sell_fee_quote = gross_sell * sell_fee
                net_sell = gross_sell - sell_fee_quote
                cycle_profit = net_sell - cost

                cash_before = cash
                btc_before = open_btc

                buy_budget -= cost
                cash -= cost
                open_btc += base_amount
                total_buy_fee_btc += buy_fee_base
                total_buy_fee_usdt_equiv += buy_fee_base * buy_price

                position_id += 1
                pos = {
                    "position_id": position_id,
                    "regime_id": regime["regime_id"],
                    "reference_price": regime["reference_price"],
                    "buy_time": timestamp,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "cost": cost,
                    "base_amount": base_amount,
                    "buy_fee_base": buy_fee_base,
                    "sell_fee_quote": sell_fee_quote,
                    "net_sell": net_sell,
                    "profit": cycle_profit,
                    "is_open": True,
                }
                positions[position_id] = pos
                open_by_buy_price[buy_price] = position_id
                heapq.heappush(sell_heap, (sell_price, position_id))
                event_id += 1

                trade_events.append({
                    "event_id": event_id,
                    "time": timestamp,
                    "side": "BUY",
                    "position_id": position_id,
                    "regime_id": regime["regime_id"],
                    "reference_price": regime["reference_price"],
                    "price": buy_price,
                    "base_amount": base_amount,
                    "quote_amount": cost,
                    "fee_base": buy_fee_base,
                    "fee_quote": 0.0,
                    "realized_profit": 0.0,
                    "cash_movement": -cost,
                    "grid_cashflow": 0.0,
                    "cash_before": cash_before,
                    "cash_after": cash,
                    "btc_before": btc_before,
                    "btc_after": open_btc,
                })

        # 3) Mark portfolio to market at candle Close.
        equity_values[i] = cash + open_btc * close_price
        cash_values[i] = cash
        btc_values[i] = open_btc
        reference_values[i] = regime["reference_price"]
        regime_values[i] = regime["regime_id"]

        # 4) Causal re-center: decision uses this candle Close and becomes
        # effective on the next candle. Old positions are not modified.
        trigger_distance = recenter_trigger_grids * gap
        if (
            close_price >= regime["reference_price"] + trigger_distance
            or close_price <= regime["reference_price"] - trigger_distance
        ):
            new_reference = round_to_gap(close_price, gap)

            if new_reference != regime["reference_price"]:
                old_regime = regime
                regime_id += 1
                regime = build_dynamic_regime(
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

                recenter_events.append({
                    "decision_time": timestamp,
                    "direction": direction,
                    "close": close_price,
                    "old_regime_id": old_regime["regime_id"],
                    "new_regime_id": regime_id,
                    "old_reference": old_regime["reference_price"],
                    "new_reference": new_reference,
                    "old_floor": old_regime["floor"],
                    "old_ceiling": old_regime["ceiling"],
                    "new_floor": regime["floor"],
                    "new_ceiling": regime["ceiling"],
                })
                regime_history.append({
                    "regime_id": regime_id,
                    "effective_time": (
                        data.iloc[i + 1]["open_time"] if i + 1 < n_rows else pd.NaT
                    ),
                    "reference_price": regime["reference_price"],
                    "floor": regime["floor"],
                    "ceiling": regime["ceiling"],
                    "reason": f"RECENTER_{direction}",
                })

        prev_close = close_price

    equity_curve = pd.DataFrame({
        "open_time": data["open_time"].to_numpy(),
        "close": data["close"].to_numpy(float),
        "cash": cash_values,
        "btc": btc_values,
        "equity": equity_values,
        "reference_price": reference_values,
        "regime_id": regime_values,
    })

    running_peak = np.maximum.accumulate(equity_values)
    drawdown = equity_values / running_peak - 1.0
    equity_curve["drawdown"] = drawdown

    max_drawdown = float(drawdown.min())
    final_equity = float(equity_values[-1])
    net_return = final_equity / initial_capital - 1.0

    elapsed_days = (
        data["open_time"].iloc[-1] - data["open_time"].iloc[0]
    ).total_seconds() / 86400.0

    annualized_return = np.nan
    if elapsed_days > 0 and final_equity > 0:
        annualized_log_growth = (
            np.log(final_equity / initial_capital) * (365.25 / elapsed_days)
        )
        if annualized_log_growth < 700:
            annualized_return = float(np.expm1(annualized_log_growth))

    calmar_ratio = np.nan
    if max_drawdown < 0 and np.isfinite(annualized_return):
        calmar_ratio = float(annualized_return / abs(max_drawdown))

    trade_log = pd.DataFrame(trade_events)
    if not trade_log.empty:
        trade_log["cumulative_cash_movement"] = (
            trade_log["cash_movement"].cumsum()
        )
        trade_log["cumulative_grid_cashflow"] = (
            trade_log["grid_cashflow"].cumsum()
        )

    open_positions = [p for p in positions.values() if p["is_open"]]

    summary = {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "net_return": float(net_return),
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "completed_cycles": int(completed_cycles),
        "open_positions": int(len(open_positions)),
        "final_cash": float(cash),
        "final_btc": float(open_btc),
        "realized_profit": float(realized_profit),
        "unrealized_pnl": float(
            final_equity - initial_capital - realized_profit
        ),
        "buy_fee_btc": float(total_buy_fee_btc),
        "buy_fee_usdt_equiv": float(total_buy_fee_usdt_equiv),
        "sell_fee_usdt": float(total_sell_fee_usdt),
        "total_fee_usdt_equiv": float(
            total_buy_fee_usdt_equiv + total_sell_fee_usdt
        ),
        "recenter_count": int(len(recenter_events)),
    }

    return {
        "summary": summary,
        "trade_log": trade_log,
        "completed_trades": pd.DataFrame(completed_trades),
        "equity_curve": equity_curve,
        "open_positions": pd.DataFrame(open_positions),
        "recenter_log": pd.DataFrame(recenter_events),
        "regime_history": pd.DataFrame(regime_history),
    }
