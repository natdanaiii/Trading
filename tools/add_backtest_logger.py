import json
from pathlib import Path

NOTEBOOK_PATH = Path('Grid_trading.ipynb')
LOGGER_TITLE = '## 12. Export Latest Run Log to GitHub'

LOGGER_MARKDOWN = '''## 12. Export Latest Run Log to GitHub

This cell builds a compact machine-readable report for review and updates:

`logs/latest_backtest_log.json`

The GitHub token is read from **Colab Secrets** as `GITHUB_TOKEN`; it is never stored in this notebook. Use a fine-grained GitHub token with **Contents: Read and write** permission for the `natdanaiii/Trading` repository.

If the secret has not been configured yet, the JSON is still saved locally as `/content/latest_backtest_log.json`, and the GitHub upload is skipped with a setup message.
'''

LOGGER_CODE = r'''import base64
import json
from datetime import datetime, timezone

import requests

GITHUB_REPO = 'natdanaiii/Trading'
GITHUB_BRANCH = 'main'
GITHUB_LOG_PATH = 'logs/latest_backtest_log.json'
LOCAL_LOG_PATH = '/content/latest_backtest_log.json'
GITHUB_SECRET_NAME = 'GITHUB_TOKEN'

def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient='records'))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_list())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)

try:
    from google.colab import userdata
    github_token = userdata.get(GITHUB_SECRET_NAME)
except Exception:
    github_token = None

github_headers = {
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}
if github_token:
    github_headers['Authorization'] = f'Bearer {github_token}'

notebook_api_url = (
    'https://api.github.com/repos/'
    f'{GITHUB_REPO}/contents/Grid_trading.ipynb'
)
commit_api_url = (
    'https://api.github.com/repos/'
    f'{GITHUB_REPO}/commits/{GITHUB_BRANCH}'
)

source_notebook_blob_sha = None
source_main_commit_sha = None
try:
    notebook_meta_response = requests.get(
        notebook_api_url,
        headers=github_headers,
        params={'ref': GITHUB_BRANCH},
        timeout=30,
    )
    if notebook_meta_response.ok:
        source_notebook_blob_sha = notebook_meta_response.json().get('sha')

    commit_meta_response = requests.get(
        commit_api_url,
        headers=github_headers,
        timeout=30,
    )
    if commit_meta_response.ok:
        source_main_commit_sha = commit_meta_response.json().get('sha')
except requests.RequestException:
    pass

trade_side = (
    df_trade_log['side']
    if 'side' in df_trade_log.columns
    else pd.Series(dtype='object')
)

open_grid_columns = [
    'level', 'buy_price', 'sell_price', 'capital_per_level',
    'base_amount', 'holding', 'buy_time',
]
open_grid_positions = df_grid_state.loc[
    df_grid_state['holding'],
    [col for col in open_grid_columns if col in df_grid_state.columns],
]

verification_ok = bool(
    df_verification_report['result'].eq('PASS').all()
)
system_audit_ok = bool(
    df_system_test_log['status'].eq('PASS').all()
)

run_log = {
    'log_schema_version': 1,
    'run_info': {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'repository': GITHUB_REPO,
        'branch': GITHUB_BRANCH,
        'source_main_commit_sha': source_main_commit_sha,
        'source_notebook_blob_sha': source_notebook_blob_sha,
        'symbol': SYMBOL,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'timeframe': '1m',
        'data_rows': int(len(df_1m)),
        'data_first_time': df_1m['open_time'].iloc[0] if len(df_1m) else None,
        'data_last_time': df_1m['open_time'].iloc[-1] if len(df_1m) else None,
    },
    'parameters': {
        'backtest_capital': BACKTEST_CAPITAL,
        'backtest_floor': BACKTEST_FLOOR,
        'backtest_ceiling': BACKTEST_CEILING,
        'backtest_gap': BACKTEST_GAP,
        'price_rounding': PRICE_ROUNDING,
        'number_of_grids': NUMBER_OF_GRIDS,
        'capital_per_grid': CAPITAL_PER_LEVEL,
        'buy_fee': BUY_FEE,
        'sell_fee': SELL_FEE,
        'historical_low': historical_low,
        'historical_high': historical_high,
    },
    'verification': {
        'status': 'PASS' if verification_ok else 'FAIL',
        'scenarios': df_verification_report,
    },
    'system_audit': {
        'status': 'PASS' if system_audit_ok else 'FAIL',
        'checks': df_system_test_log,
    },
    'backtest_summary': backtest_summary,
    'reconciliation': {
        'net_cash_movement': (
            df_trade_log['cash_movement'].sum()
            if 'cash_movement' in df_trade_log.columns
            else 0.0
        ),
        'total_grid_cashflow': (
            df_trade_log['grid_cashflow'].sum()
            if 'grid_cashflow' in df_trade_log.columns
            else 0.0
        ),
        'portfolio_gain': (
            backtest_summary['final_equity']
            - backtest_summary['initial_capital']
        ),
    },
    'monthly_grid_cashflow': df_grid_cashflow_monthly,
    'monthly_portfolio_pnl': df_portfolio_pnl_monthly,
    'diagnostics': {
        'trade_event_count': int(len(df_trade_log)),
        'buy_event_count': int(trade_side.eq('BUY').sum()),
        'sell_event_count': int(trade_side.eq('SELL').sum()),
        'completed_cycle_count': int(len(df_completed_trades)),
        'open_position_count': int(backtest_summary['open_positions']),
        'first_10_trade_events': df_trade_log.head(10),
        'last_10_trade_events': df_trade_log.tail(10),
        'open_grid_positions': open_grid_positions,
    },
}

run_log_safe = _json_safe(run_log)
log_json = json.dumps(
    run_log_safe,
    indent=2,
    ensure_ascii=False,
    allow_nan=False,
)

with open(LOCAL_LOG_PATH, 'w', encoding='utf-8') as log_file:
    log_file.write(log_json)

print('===== BACKTEST LOG =====')
print(f'Local log    : {LOCAL_LOG_PATH}')
print(f'Verification : {run_log_safe["verification"]["status"]}')
print(f'System Audit : {run_log_safe["system_audit"]["status"]}')
print(f'Notebook SHA : {source_notebook_blob_sha}')

if not github_token:
    print()
    print(
        'GitHub upload SKIPPED: Colab Secret '
        f'{GITHUB_SECRET_NAME!r} was not found.'
    )
    print(
        'Add the secret once, give this notebook access, '
        'then rerun this cell.'
    )
else:
    try:
        log_api_url = (
            'https://api.github.com/repos/'
            f'{GITHUB_REPO}/contents/{GITHUB_LOG_PATH}'
        )
        existing_response = requests.get(
            log_api_url,
            headers=github_headers,
            params={'ref': GITHUB_BRANCH},
            timeout=30,
        )

        payload = {
            'message': 'Update latest backtest log [skip ci]',
            'content': base64.b64encode(
                log_json.encode('utf-8')
            ).decode('ascii'),
            'branch': GITHUB_BRANCH,
        }

        if existing_response.status_code == 200:
            payload['sha'] = existing_response.json()['sha']
        elif existing_response.status_code != 404:
            existing_response.raise_for_status()

        upload_response = requests.put(
            log_api_url,
            headers=github_headers,
            json=payload,
            timeout=30,
        )
        upload_response.raise_for_status()
        log_commit_sha = (
            upload_response.json()
            .get('commit', {})
            .get('sha')
        )

        print()
        print(f'GitHub log    : {GITHUB_LOG_PATH}')
        print('Upload        : SUCCESS')
        print(f'Log commit    : {log_commit_sha}')
    except requests.RequestException as exc:
        print()
        print(f'GitHub upload FAILED: {exc}')
        print('The local JSON log was still saved.')
'''


def source_text(cell):
    source = cell.get('source', '')
    return ''.join(source) if isinstance(source, list) else source


def make_markdown(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [text],
    }


def make_code(text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [text],
    }


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))

# Remove an older logger section if this helper is run again.
new_cells = []
skip_next_code = False
for cell in notebook['cells']:
    text = source_text(cell)
    if cell.get('cell_type') == 'markdown' and LOGGER_TITLE in text:
        skip_next_code = True
        continue
    if skip_next_code and cell.get('cell_type') == 'code':
        skip_next_code = False
        continue
    new_cells.append(cell)

notebook['cells'] = new_cells

# Update workflow line without changing strategy logic.
first_text = source_text(notebook['cells'][0])
if '**Workflow:**' in first_text and 'GitHub Log' not in first_text:
    first_text = first_text.replace(
        ' → Audit → Results',
        ' → Audit → Results → GitHub Log',
    )
    notebook['cells'][0]['source'] = [first_text]

notebook['cells'].append(make_markdown(LOGGER_MARKDOWN))
notebook['cells'].append(make_code(LOGGER_CODE))

NOTEBOOK_PATH.write_text(
    json.dumps(
        notebook,
        ensure_ascii=False,
        indent=1,
    ),
    encoding='utf-8',
)

print(f'Patched {NOTEBOOK_PATH} with automatic GitHub run logging.')
