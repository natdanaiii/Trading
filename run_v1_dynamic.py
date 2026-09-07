"""Run V0 fixed-grid baseline and V1 dynamic re-centering grid comparison in Colab."""

import os
import json
import base64
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive

# -------------------------
# 1. Setup & Data
# -------------------------
drive.mount('/content/drive')
DATA_DIR = '/content/drive/MyDrive/03.Trading/00.Live Trading'

SYMBOL = 'BTCUSDT'
START_DATE = '2024-01-01'
END_DATE = '2026-01-01'

BUY_FEE = 0.001
SELL_FEE = 0.001

BACKTEST_CAPITAL = 3000.0
BACKTEST_GAP = 1000.0
PRICE_ROUNDING = 1000.0

def load_market_data(symbol, timeframe, data_dir):
    path = os.path.join(data_dir, f'{symbol}-{timeframe}-combined.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    return (
        df.drop_duplicates('open_time')
        .sort_values('open_time')
        .reset_index(drop=True)
    )

df_1m = load_market_data(SYMBOL, '1m', DATA_DIR)

start_ts = pd.Timestamp(START_DATE, tz='UTC')
end_ts = pd.Timestamp(END_DATE, tz='UTC')
df_1m = df_1m.loc[
    (df_1m['open_time'] >= start_ts)
    & (df_1m['open_time'] < end_ts)
].reset_index(drop=True)

print('===== DATA =====')
print(f'Rows: {len(df_1m):,}')
print(f'Period: {df_1m.open_time.min()} -> {df_1m.open_time.max()}')

# -------------------------
# 2. Load current V0 engine
# -------------------------
MAIN_NOTEBOOK_URL = (
    'https://raw.githubusercontent.com/'
    'natdanaiii/Trading/main/Grid_trading.ipynb'
)
main_response = requests.get(MAIN_NOTEBOOK_URL, timeout=30)
main_response.raise_for_status()
main_nb = main_response.json()

def exec_main_cell_containing(marker):
    for cell in main_nb['cells']:
        source = ''.join(cell.get('source', []))
        if cell.get('cell_type') == 'code' and marker in source:
            exec(source, globals())
            return
    raise RuntimeError(f'Could not find main cell containing: {marker}')

exec_main_cell_containing('def build_excel_grid_table')
exec_main_cell_containing('def run_grid_backtest')
print('Loaded current V0 engine from Grid_trading.ipynb')

# -------------------------
# 3. Run V0 fixed-grid baseline
# -------------------------
historical_low = df_1m['low'].min()
historical_high = df_1m['high'].max()

BACKTEST_FLOOR = (
    np.floor(historical_low / PRICE_ROUNDING) * PRICE_ROUNDING
)
BACKTEST_CEILING = (
    np.ceil(historical_high / PRICE_ROUNDING) * PRICE_ROUNDING
)

raw_backtest_levels = (
    BACKTEST_CEILING - BACKTEST_FLOOR
) / BACKTEST_GAP

if not np.isclose(raw_backtest_levels, round(raw_backtest_levels)):
    raise ValueError('V0 backtest range must be divisible by BACKTEST_GAP.')

NUMBER_OF_GRIDS = int(round(raw_backtest_levels))
CAPITAL_PER_LEVEL = BACKTEST_CAPITAL / NUMBER_OF_GRIDS

df_grid_v0 = build_excel_grid_table(
    BACKTEST_CAPITAL,
    BACKTEST_CEILING,
    BACKTEST_FLOOR,
    BACKTEST_GAP,
    BUY_FEE,
    SELL_FEE,
)

v0 = run_grid_backtest(df_1m, df_grid_v0, BACKTEST_CAPITAL)
s0 = v0['summary']

print('\n===== V0 FIXED GRID =====')
print(f'Floor              : {BACKTEST_FLOOR:,.0f}')
print(f'Ceiling            : {BACKTEST_CEILING:,.0f}')
print(f'Gap                : {BACKTEST_GAP:,.0f}')
print(f'Number of Grids    : {NUMBER_OF_GRIDS}')
print(f'Capital / Grid     : {CAPITAL_PER_LEVEL:,.4f} USDT')
print(f'Final Equity       : {s0["final_equity"]:,.2f} USDT')
print(f'Net Return         : {s0["net_return"]:.2%}')

# -------------------------
# 4. V1 configuration
# -------------------------
V1_GRID_GAP = BACKTEST_GAP
V1_NUMBER_OF_GRIDS = 30
V1_RECENTER_TRIGGER_GRIDS = 5
V1_CAPITAL_PER_GRID = BACKTEST_CAPITAL / V1_NUMBER_OF_GRIDS

print('\n===== V1 CONFIGURATION =====')
print(f'Gap                    : {V1_GRID_GAP:,.0f} USDT')
print(f'Active Grid Intervals  : {V1_NUMBER_OF_GRIDS}')
print(f'Capital / Grid         : {V1_CAPITAL_PER_GRID:,.2f} USDT')
print(
    f'Recenter Trigger       : {V1_RECENTER_TRIGGER_GRIDS} grids '
    f'({V1_RECENTER_TRIGGER_GRIDS * V1_GRID_GAP:,.0f} USDT)'
)
print('Compounding             : OFF')

# -------------------------
# 5. Load V1 engine
# -------------------------
V1_ENGINE_URL = (
    'https://raw.githubusercontent.com/'
    'natdanaiii/Trading/main/dynamic_grid_v1.py'
)
v1_engine_response = requests.get(V1_ENGINE_URL, timeout=30)
v1_engine_response.raise_for_status()
exec(v1_engine_response.text, globals())
print('Loaded dynamic_grid_v1.py')

# -------------------------
# 6. Deterministic V1 verification
# -------------------------
synthetic = pd.DataFrame({
    'open_time': pd.date_range(
        '2024-01-01', periods=2, freq='min', tz='UTC'
    ),
    'open': [100.0, 90.0],
    'high': [100.0, 100.0],
    'low': [89.0, 89.0],
    'close': [90.0, 90.0],
})

verify_v1 = run_dynamic_grid_backtest(
    synthetic,
    initial_capital=400.0,
    gap=10.0,
    number_of_grids=4,
    recenter_trigger_grids=1,
    buy_fee=0.001,
    sell_fee=0.001,
)

verify_completed = verify_v1['completed_trades']
verify_recenter = verify_v1['recenter_log']

verification_rows = [
    {
        'Scenario': 'Recenter occurred',
        'Expected': '>= 1',
        'Actual': str(len(verify_recenter)),
        'Result': 'PASS' if len(verify_recenter) >= 1 else 'FAIL',
    },
    {
        'Scenario': 'Old regime position preserved',
        'Expected': 'regime_id = 0',
        'Actual': (
            f'regime_id = {int(verify_completed.iloc[0]["regime_id"])}'
            if len(verify_completed) else 'no completed trade'
        ),
        'Result': (
            'PASS'
            if len(verify_completed)
            and int(verify_completed.iloc[0]['regime_id']) == 0
            else 'FAIL'
        ),
    },
    {
        'Scenario': 'Original SELL target preserved',
        'Expected': '90 -> 100',
        'Actual': (
            f'{verify_completed.iloc[0]["buy_price"]:.0f} -> '
            f'{verify_completed.iloc[0]["sell_price"]:.0f}'
            if len(verify_completed) else 'no completed trade'
        ),
        'Result': (
            'PASS'
            if len(verify_completed)
            and np.isclose(verify_completed.iloc[0]['buy_price'], 90.0)
            and np.isclose(verify_completed.iloc[0]['sell_price'], 100.0)
            else 'FAIL'
        ),
    },
]
df_v1_verification = pd.DataFrame(verification_rows)

print('\n===== V1 VERIFICATION =====')
display(df_v1_verification)

if not (df_v1_verification['Result'] == 'PASS').all():
    raise AssertionError('V1 VERIFICATION FAILED')
print('V1 VERIFICATION: PASS')

# -------------------------
# 7. Run historical V1
# -------------------------
v1 = run_dynamic_grid_backtest(
    df_1m,
    initial_capital=BACKTEST_CAPITAL,
    gap=V1_GRID_GAP,
    number_of_grids=V1_NUMBER_OF_GRIDS,
    recenter_trigger_grids=V1_RECENTER_TRIGGER_GRIDS,
    buy_fee=BUY_FEE,
    sell_fee=SELL_FEE,
)
s1 = v1['summary']

print('\n===== V1 DYNAMIC GRID SUMMARY =====')
print(f'Initial Capital     : {s1["initial_capital"]:,.2f} USDT')
print(f'Final Equity        : {s1["final_equity"]:,.2f} USDT')
print(f'Net Return          : {s1["net_return"]:.2%}')
print(f'Annualized Return   : {s1["annualized_return"]:.2%}')
print(f'Max Drawdown        : {s1["max_drawdown"]:.2%}')
print(f'Calmar Ratio        : {s1["calmar_ratio"]:.3f}')
print(f'Completed Cycles    : {s1["completed_cycles"]:,}')
print(f'Open Positions      : {s1["open_positions"]:,}')
print(f'Recenter Count      : {s1["recenter_count"]:,}')
print(f'Final Cash          : {s1["final_cash"]:,.2f} USDT')
print(f'Final BTC           : {s1["final_btc"]:.8f} BTC')
print(f'Realized Profit     : {s1["realized_profit"]:,.2f} USDT')
print(f'Unrealized P&L      : {s1["unrealized_pnl"]:,.2f} USDT')
print(f'Total Fee           : {s1["total_fee_usdt_equiv"]:,.2f} USDT')

# -------------------------
# 8. V0 vs V1 comparison
# -------------------------
comparison_specs = [
    ('Final Equity (USDT)', 'final_equity', 1.0),
    ('Net Return (%)', 'net_return', 100.0),
    ('Annualized Return (%)', 'annualized_return', 100.0),
    ('Max Drawdown (%)', 'max_drawdown', 100.0),
    ('Calmar Ratio', 'calmar_ratio', 1.0),
    ('Completed Cycles', 'completed_cycles', 1.0),
    ('Open Positions', 'open_positions', 1.0),
    ('Realized Profit (USDT)', 'realized_profit', 1.0),
    ('Unrealized P&L (USDT)', 'unrealized_pnl', 1.0),
    ('Total Fees (USDT)', 'total_fee_usdt_equiv', 1.0),
    ('Final Cash (USDT)', 'final_cash', 1.0),
    ('Final BTC', 'final_btc', 1.0),
]

comparison_rows = []
for label, key, scale in comparison_specs:
    v0_value = s0[key] * scale
    v1_value = s1[key] * scale
    comparison_rows.append({
        'Metric': label,
        'V0 Fixed Grid': v0_value,
        'V1 Dynamic Grid': v1_value,
        'Difference (V1-V0)': v1_value - v0_value,
    })

comparison_rows.append({
    'Metric': 'Recenter Count',
    'V0 Fixed Grid': 0,
    'V1 Dynamic Grid': s1['recenter_count'],
    'Difference (V1-V0)': s1['recenter_count'],
})

df_comparison = pd.DataFrame(comparison_rows)
print('\n===== V0 vs V1 COMPARISON =====')
display(df_comparison.round(6))

print(
    '\nComparison note: V1 uses 30 active intervals x 100 USDT, '
    'while V0 uses 89 intervals x ~33.71 USDT. '
    'This is a strategy comparison, not yet a one-variable causal experiment.'
)

# -------------------------
# 9. Monthly comparison
# -------------------------
def monthly_portfolio(result, initial_capital, prefix):
    eq = result['equity_curve'][['open_time', 'equity']].copy()
    eq['month'] = eq['open_time'].dt.strftime('%Y-%m')

    monthly = (
        eq.groupby('month', as_index=False)['equity']
        .last()
        .rename(columns={'equity': f'{prefix}_ending_equity'})
    )

    beginning = monthly[f'{prefix}_ending_equity'].shift(1)
    beginning.iloc[0] = initial_capital
    monthly[f'{prefix}_net_pnl'] = (
        monthly[f'{prefix}_ending_equity'] - beginning
    )
    return monthly

v0_monthly = monthly_portfolio(v0, BACKTEST_CAPITAL, 'v0')
v1_monthly = monthly_portfolio(v1, BACKTEST_CAPITAL, 'v1')
df_monthly_comparison = v0_monthly.merge(v1_monthly, on='month')

df_monthly_comparison['equity_difference'] = (
    df_monthly_comparison['v1_ending_equity']
    - df_monthly_comparison['v0_ending_equity']
)
df_monthly_comparison['net_pnl_difference'] = (
    df_monthly_comparison['v1_net_pnl']
    - df_monthly_comparison['v0_net_pnl']
)

print('\n===== MONTHLY COMPARISON =====')
display(df_monthly_comparison.round(2))

plot_df = df_monthly_comparison.copy()
plot_df['month_date'] = pd.to_datetime(plot_df['month'])

plt.figure(figsize=(12, 5))
plt.plot(
    plot_df['month_date'],
    plot_df['v0_ending_equity'],
    label='V0 Fixed Grid',
)
plt.plot(
    plot_df['month_date'],
    plot_df['v1_ending_equity'],
    label='V1 Dynamic Grid',
)
plt.title('BTC Spot Grid - V0 vs V1 Month-End Equity')
plt.xlabel('Month')
plt.ylabel('Equity (USDT)')
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# -------------------------
# 10. Recenter diagnostics
# -------------------------
print('\n===== V1 RECENTER DIAGNOSTICS =====')
print(f'Total re-centers: {len(v1["recenter_log"]):,}')

if len(v1['recenter_log']):
    print('\nFirst 10 re-centers')
    display(v1['recenter_log'].head(10))
    print('\nLast 10 re-centers')
    display(v1['recenter_log'].tail(10))

print('\nLatest regime state')
display(v1['regime_history'].tail(10))

# -------------------------
# 11. V1 audit
# -------------------------
trade_log_v1 = v1['trade_log']
equity_v1 = v1['equity_curve']

cash_movement_sum = (
    trade_log_v1['cash_movement'].sum()
    if len(trade_log_v1) else 0.0
)
grid_cashflow_sum = (
    trade_log_v1['grid_cashflow'].sum()
    if len(trade_log_v1) else 0.0
)

cash_error = abs(
    BACKTEST_CAPITAL + cash_movement_sum - s1['final_cash']
)
realized_error = abs(
    grid_cashflow_sum - s1['realized_profit']
)
equity_error = np.max(np.abs(
    equity_v1['cash']
    + equity_v1['btc'] * equity_v1['close']
    - equity_v1['equity']
))
minimum_cash = equity_v1['cash'].min()

if len(v1['recenter_log']):
    recenter_distance_ok = (
        (
            v1['recenter_log']['close']
            - v1['recenter_log']['old_reference']
        ).abs()
        + 1e-9
        >= V1_RECENTER_TRIGGER_GRIDS * V1_GRID_GAP
    ).all()
else:
    recenter_distance_ok = True

audit_rows = [
    {
        'Test': 'Cash reconciliation',
        'Expected': '<= 1e-8',
        'Actual': cash_error,
        'Status': 'PASS' if cash_error <= 1e-8 else 'FAIL',
    },
    {
        'Test': 'Grid Cashflow = realized profit',
        'Expected': '<= 1e-8',
        'Actual': realized_error,
        'Status': 'PASS' if realized_error <= 1e-8 else 'FAIL',
    },
    {
        'Test': 'Cash + BTC x Close = Equity',
        'Expected': '<= 1e-8',
        'Actual': equity_error,
        'Status': 'PASS' if equity_error <= 1e-8 else 'FAIL',
    },
    {
        'Test': 'Cash never negative',
        'Expected': '>= 0',
        'Actual': minimum_cash,
        'Status': 'PASS' if minimum_cash >= -1e-8 else 'FAIL',
    },
    {
        'Test': 'Recenter threshold respected',
        'Expected': 'True',
        'Actual': bool(recenter_distance_ok),
        'Status': 'PASS' if recenter_distance_ok else 'FAIL',
    },
]

df_v1_audit = pd.DataFrame(audit_rows)
print('\n===== V1 SYSTEM AUDIT =====')
display(df_v1_audit)

V1_AUDIT_STATUS = (
    'PASS' if (df_v1_audit['Status'] == 'PASS').all() else 'FAIL'
)
print(f'V1 SYSTEM AUDIT: {V1_AUDIT_STATUS}')

if V1_AUDIT_STATUS != 'PASS':
    raise AssertionError('V1 SYSTEM AUDIT FAILED')

# -------------------------
# 12. Export latest V1 log
# -------------------------
def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value

def records_safe(df):
    return [
        {k: json_safe(v) for k, v in row.items()}
        for row in df.to_dict('records')
    ]

log_payload = {
    'log_schema_version': 1,
    'strategy_version': 'V1 Dynamic Re-centering Grid',
    'run_info': {
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'repository': 'natdanaiii/Trading',
        'branch': 'main',
        'notebook': 'Grid_trading_V1_dynamic.ipynb',
        'symbol': SYMBOL,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'timeframe': '1m',
        'data_rows': int(len(df_1m)),
        'data_first_time': df_1m['open_time'].min().isoformat(),
        'data_last_time': df_1m['open_time'].max().isoformat(),
    },
    'parameters': {
        'initial_capital': BACKTEST_CAPITAL,
        'buy_fee': BUY_FEE,
        'sell_fee': SELL_FEE,
        'v0_gap': BACKTEST_GAP,
        'v0_floor': BACKTEST_FLOOR,
        'v0_ceiling': BACKTEST_CEILING,
        'v0_number_of_grids': NUMBER_OF_GRIDS,
        'v0_capital_per_grid': CAPITAL_PER_LEVEL,
        'v1_gap': V1_GRID_GAP,
        'v1_number_of_grids': V1_NUMBER_OF_GRIDS,
        'v1_capital_per_grid': V1_CAPITAL_PER_GRID,
        'v1_recenter_trigger_grids': V1_RECENTER_TRIGGER_GRIDS,
        'v1_recenter_trigger_usdt': (
            V1_RECENTER_TRIGGER_GRIDS * V1_GRID_GAP
        ),
        'compounding': False,
    },
    'verification': {
        'status': 'PASS',
        'scenarios': records_safe(df_v1_verification),
    },
    'system_audit': {
        'status': V1_AUDIT_STATUS,
        'checks': records_safe(df_v1_audit),
    },
    'v0_summary': {
        k: json_safe(v) for k, v in s0.items()
    },
    'v1_summary': {
        k: json_safe(v) for k, v in s1.items()
    },
    'comparison': records_safe(df_comparison),
    'monthly_comparison': records_safe(df_monthly_comparison),
    'recenter_diagnostics': {
        'count': int(len(v1['recenter_log'])),
        'first_10': records_safe(v1['recenter_log'].head(10)),
        'last_10': records_safe(v1['recenter_log'].tail(10)),
    },
}

LOCAL_LOG_PATH = '/content/latest_v1_dynamic_backtest_log.json'
with open(LOCAL_LOG_PATH, 'w') as f:
    json.dump(log_payload, f, indent=2)

print('\n===== V1 BACKTEST LOG =====')
print(f'Local log      : {LOCAL_LOG_PATH}')
print('Verification   : PASS')
print(f'System Audit   : {V1_AUDIT_STATUS}')

try:
    from google.colab import userdata
    github_token = userdata.get('GITHUB_TOKEN')
except Exception:
    github_token = None

if not github_token:
    print()
    print("GitHub upload SKIPPED: Colab Secret 'GITHUB_TOKEN' was not found.")
else:
    repo = 'natdanaiii/Trading'
    path = 'logs/latest_v1_dynamic_backtest_log.json'
    api_url = f'https://api.github.com/repos/{repo}/contents/{path}'

    headers = {
        'Authorization': f'Bearer {github_token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }

    existing = requests.get(api_url, headers=headers, timeout=30)
    existing_sha = (
        existing.json().get('sha')
        if existing.status_code == 200
        else None
    )

    encoded = base64.b64encode(
        json.dumps(log_payload, indent=2).encode()
    ).decode()

    body = {
        'message': 'Update latest V1 dynamic grid backtest log',
        'content': encoded,
        'branch': 'main',
    }
    if existing_sha:
        body['sha'] = existing_sha

    upload = requests.put(
        api_url,
        headers=headers,
        json=body,
        timeout=30,
    )
    upload.raise_for_status()

    upload_json = upload.json()
    print()
    print('GitHub upload   : SUCCESS')
    print(f'Path            : {path}')
    print(f'Commit SHA      : {upload_json["commit"]["sha"]}')
