import hashlib
import json
import re
from pathlib import Path

NOTEBOOK = Path("Grid_trading_V0.ipynb")


def get_source(cell):
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def set_source(cell, text):
    cell["source"] = text


def find_markdown(cells, text):
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown" and text in get_source(cell):
            return index
    raise RuntimeError(f"Markdown cell not found: {text}")


def next_code_cell(cells, start_index):
    for index in range(start_index + 1, len(cells)):
        if cells[index].get("cell_type") == "code":
            return index
    raise RuntimeError(f"No code cell found after index {start_index}")


def section_hash(cells):
    payload = json.dumps(cells, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = nb["cells"]

section_2_index = find_markdown(cells, "# 2. Backtest System")
trading_system_before = section_hash(cells[:section_2_index])

# -----------------------------------------------------------------------------
# Section 2.2: replace only the benchmark function. Data loading and performance
# metrics remain unchanged.
# -----------------------------------------------------------------------------
section_22_index = find_markdown(cells, "## 2.2 Data, Metrics & Buy/Hold Benchmark")
set_source(cells[section_22_index], "## 2.2 Data, Metrics & Monthly DCA Benchmark\n")
section_22_code_index = next_code_cell(cells, section_22_index)
section_22_source = get_source(cells[section_22_code_index])

marker = "def build_buy_hold_benchmark"
if marker not in section_22_source:
    raise RuntimeError("Buy & Hold benchmark function was not found.")

section_22_prefix = section_22_source.split(marker, 1)[0]

dca_function = '''def build_monthly_dca_benchmark(data, initial_capital, buy_fee):
    benchmark_data = data.sort_values("open_time").reset_index(drop=True).copy()

    expected_months = (
        benchmark_data
        .assign(
            year=benchmark_data["open_time"].dt.year,
            month=benchmark_data["open_time"].dt.month,
        )[["year", "month"]]
        .drop_duplicates()
    )

    day_one_data = benchmark_data.loc[
        benchmark_data["open_time"].dt.day.eq(1)
    ].copy()

    monthly_buy_rows = (
        day_one_data
        .assign(
            year=day_one_data["open_time"].dt.year,
            month=day_one_data["open_time"].dt.month,
        )
        .groupby(["year", "month"], sort=True)
        .head(1)
    )

    number_of_buys = len(monthly_buy_rows)
    if number_of_buys == 0:
        raise ValueError("No monthly DCA purchase dates were found.")

    if number_of_buys != len(expected_months):
        raise ValueError(
            "Missing first-day market data for one or more DCA months."
        )

    dca_amount_usdt = initial_capital / number_of_buys

    btc_additions = np.zeros(len(benchmark_data), dtype=float)
    cash_spending = np.zeros(len(benchmark_data), dtype=float)
    schedule_rows = []

    for row_index, row in monthly_buy_rows.iterrows():
        buy_price = float(row["open"])
        gross_btc = dca_amount_usdt / buy_price
        buy_fee_btc = gross_btc * buy_fee
        net_btc = gross_btc - buy_fee_btc

        btc_additions[row_index] = net_btc
        cash_spending[row_index] = dca_amount_usdt

        schedule_rows.append({
            "buy_time": row["open_time"],
            "buy_price": buy_price,
            "dca_amount_usdt": float(dca_amount_usdt),
            "buy_fee_usdt_equiv": float(buy_fee_btc * buy_price),
            "net_btc": float(net_btc),
        })

    btc_curve = np.cumsum(btc_additions)
    cash_curve = initial_capital - np.cumsum(cash_spending)
    cash_curve[np.abs(cash_curve) < 1e-10] = 0.0

    equity_curve = pd.DataFrame({
        "open_time": benchmark_data["open_time"],
        "close": benchmark_data["close"],
        "cash": cash_curve,
        "btc": btc_curve,
        "equity": (
            cash_curve
            + btc_curve * benchmark_data["close"].to_numpy(float)
        ),
    })

    stats = performance_stats(
        benchmark_data,
        equity_curve,
        initial_capital,
    )

    purchase_schedule = pd.DataFrame(schedule_rows)

    return {
        "number_of_buys": int(number_of_buys),
        "dca_amount_usdt": float(dca_amount_usdt),
        "total_invested_usdt": float(
            dca_amount_usdt * number_of_buys
        ),
        "total_buy_fee_usdt_equiv": float(
            purchase_schedule["buy_fee_usdt_equiv"].sum()
        ),
        "final_cash": float(cash_curve[-1]),
        "final_btc": float(btc_curve[-1]),
        "first_buy_time": purchase_schedule["buy_time"].iloc[0],
        "last_buy_time": purchase_schedule["buy_time"].iloc[-1],
        "purchase_schedule": purchase_schedule,
        **stats,
    }
'''

set_source(cells[section_22_code_index], section_22_prefix + dca_function)

# -----------------------------------------------------------------------------
# Section 2.4: keep the V0-B backtest unchanged and swap only the external
# benchmark/comparison from Buy & Hold to Monthly DCA.
# -----------------------------------------------------------------------------
section_24_index = find_markdown(cells, "## 2.4 Run Backtest & Review Results")
section_24_code_index = next_code_cell(cells, section_24_index)
run_source = get_source(cells[section_24_code_index])

old_call = '''buy_hold = build_buy_hold_benchmark(
    df_1m,
    INITIAL_CAPITAL,
    BUY_FEE,
)
'''
new_call = '''monthly_dca = build_monthly_dca_benchmark(
    df_1m,
    INITIAL_CAPITAL,
    BUY_FEE,
)
'''
if old_call not in run_source:
    raise RuntimeError("Buy & Hold benchmark call was not found.")
run_source = run_source.replace(old_call, new_call, 1)

benchmark_block = '''benchmark_summary = {
    key: monthly_dca[key]
    for key in [
        "number_of_buys",
        "dca_amount_usdt",
        "total_invested_usdt",
        "total_buy_fee_usdt_equiv",
        "final_cash",
        "final_btc",
        "first_buy_time",
        "last_buy_time",
        "final_equity",
        "net_return",
        "annualized_return",
        "max_drawdown",
        "calmar_ratio",
    ]
}

comparison_vs_monthly_dca = {'''

run_source, replacements = re.subn(
    r'benchmark_summary = \{.*?\n\}\n\ncomparison_vs_buy_hold = \{',
    benchmark_block,
    run_source,
    count=1,
    flags=re.S,
)
if replacements != 1:
    raise RuntimeError("Benchmark summary/comparison block replacement failed.")

run_source = run_source.replace(
    "comparison_vs_buy_hold",
    "comparison_vs_monthly_dca",
)
run_source = run_source.replace(
    '"BTC Buy & Hold"',
    '"BTC Monthly DCA"',
)
run_source = run_source.replace(
    'print("\\n===== V0-B vs BTC BUY & HOLD =====")',
    'print("\\n===== MONTHLY DCA BENCHMARK =====")\n'
    'for key, value in benchmark_summary.items():\n'
    '    print(f"{key:32s}: {value}")\n\n'
    'print("\\n===== V0-B vs BTC MONTHLY DCA =====")',
)
run_source = run_source.replace(
    'f"Excess Return vs Buy & Hold : "',
    'f"Excess Return vs Monthly DCA : "',
)

set_source(cells[section_24_code_index], run_source)

# -----------------------------------------------------------------------------
# Section 3.1: only rename the logged benchmark fields and bump the schema.
# -----------------------------------------------------------------------------
section_31_index = find_markdown(cells, "## 3.1 Run Log")
section_31_code_index = next_code_cell(cells, section_31_index)
log_source = get_source(cells[section_31_code_index])

log_source = log_source.replace(
    '"log_schema_version": 5,',
    '"log_schema_version": 6,',
    1,
)
log_source = log_source.replace(
    '"buy_hold_benchmark": benchmark_summary,',
    '"monthly_dca_benchmark": benchmark_summary,',
    1,
)
log_source = log_source.replace(
    '"comparison_vs_buy_hold": comparison_vs_buy_hold,',
    '"comparison_vs_monthly_dca": comparison_vs_monthly_dca,',
    1,
)
set_source(cells[section_31_code_index], log_source)

# Guardrail: Section 1 / trading logic must be byte-for-byte identical at the
# cell-object level to the clean V0-B notebook before this benchmark change.
trading_system_after = section_hash(cells[:section_2_index])
if trading_system_before != trading_system_after:
    raise AssertionError("V0-B trading-system cells changed unexpectedly.")

# Sanity checks for the benchmark migration.
all_sources = "\n".join(get_source(cell) for cell in cells)
if "build_buy_hold_benchmark" in all_sources:
    raise AssertionError("Old Buy & Hold benchmark function still exists.")
if "comparison_vs_buy_hold" in all_sources:
    raise AssertionError("Old Buy & Hold comparison variable still exists.")
if "build_monthly_dca_benchmark" not in all_sources:
    raise AssertionError("Monthly DCA benchmark function is missing.")

# Syntax-check every Python cell before saving.
for index, cell in enumerate(cells):
    if cell.get("cell_type") == "code":
        compile(get_source(cell), f"<cell {index}>", "exec")

NOTEBOOK.write_text(
    json.dumps(nb, ensure_ascii=False, separators=(",", ":")),
    encoding="utf-8",
)

print("V0-B trading logic: unchanged")
print("Monthly DCA benchmark: applied")
print("Notebook syntax: PASS")
