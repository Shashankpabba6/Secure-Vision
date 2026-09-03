# Worker Specs — trading-bot-build (paste-ready execution prompts)

> **How to use:** This repo is locked to planner mode here, so these are the exact prompts a
> coding agent runs in an environment that CAN write files. Run waves in dependency order:
> Wave 1 (T1→T3) → Wave 2 (T4–T7) → Wave 3 (T8–T10) → Wave 4 (T11–T12) → Wave 5 (T13) → Final wave (F1–F4).
> Each block is one self-contained task. Paste the whole block as a single prompt to the worker.
> Environment: Python 3.12, `pip`, `.env` for keys (all tests use mocks when keys absent).
> **TA-Lib is optional** — prefer `pandas-ta`; if `ta` (TA-Lib) fails to build, skip it and note in the artifact.
> Do NOT touch existing files (`final_app.py`, `physiofusion/`, `utils.py`, `deepfake_*`).

Companion files: `worker-specs-2.md` (Wave 2), `worker-specs-3.md` (Wave 3), `worker-specs-4.md` (Wave 4–5 + final).

---

## Wave 1 — Foundation

### W1-T1 — Scaffold `trading/` package + dependency manifest
```
TASK: Create the trading bot Python package skeleton inside this repo (WebApp/).
DELIVERABLE:
  - trading/__init__.py
  - trading/{config,data,agents,broker,orchestration,monitor,backtest,ui}/__init__.py (each empty or minimal)
  - requirements.txt listing: langgraph, langchain, langchain-openai, tradingagents, alpaca-py,
    pandas, pandas-ta, yfinance, python-dotenv, pydantic-settings, apscheduler, python-telegram-bot,
    backtrader, streamlit, matplotlib, mplfinance, pytest, httpx
  - pyproject.toml (optional, same deps) OR keep requirements.txt only
  - .gitignore appending: .env, .omo/, __pycache__/, *.db, trading/state/
  - tests/ directory with tests/__init__.py and tests/conftest.py (empty fixtures ok)
SCOPE: WebApp/ only. Do NOT modify existing files.
BASELINE TEST (write first, passes on unchanged code): tests/test_smoke.py with `def test_repo_importable(): assert True`
FAILING-FIRST PROOF: after creating package, tests/test_import.py::test_trading_imports asserts `import trading` succeeds.
IMPLEMENTATION CONSTRAINTS: keep deps in one manifest; do not pin conflicting versions with the existing repo.
AUTOMATED VERIFICATION: `pip install -r requirements.txt` (allow TA-Lib to be skipped if it fails to build, document it),
  then `python -c "import trading"` and `pytest --collect-only -q` (0 errors).
MANUAL QA: open a shell, `cd WebApp && python -c "import trading; print('ok')"` prints ok.
ADVERSARIAL CLASSES: dirty_worktree — must not edit files outside trading/ and tests/; hung_commands — wrap pip in a 300s timeout.
ARTIFACT: report created file tree + which deps installed vs skipped.
```

### W1-T2 — Typed config & secrets module (paper/live mode + model selection)
```
TASK: Implement trading/config.py with pydantic-settings loading .env.
DELIVERABLE: trading/config.py exposing `settings` with fields:
  MODE: Literal["paper","live"] = "paper"
  ASSET_UNIVERSE: list[str] (from env CSV or watchlist.txt)
  DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL="https://api.deepseek.com"
  MINIMAX_API_KEY, MINIMAX_BASE_URL (OpenAI-compatible)
  ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL="https://paper-api.alpaca.markets"
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  MODEL_QUICK="deepseek-v4-flash", MODEL_REASON="deepseek-v4-pro", MODEL_VISION="minimax-m3"
  KILL_SWITCH: int = 0
  Add a `.env.example` (no real values) and `watchlist.txt` (e.g. AAPL,MSFT,NVDA).
SCOPE: only trading/config.py, .env.example, watchlist.txt.
BASELINE TEST: tests/test_config.py::test_defaults — settings.MODE == "paper" when no .env present (use monkeypatch).
FAILING-FIRST PROOF: tests/test_config.py::test_missing_required_raises — deleting DEEPSEEK_API_KEY raises ValidationError naming the field.
IMPLEMENTATION CONSTRAINTS: never hardcode keys; never commit .env; validate on import.
AUTOMATED VERIFICATION: `pytest tests/test_config.py -q` all pass; `python -c "from trading.config import settings; assert settings.MODE in ('paper','live')"`.
MANUAL QA: create a temp .env with fake keys, run import, assert settings.MODEL_REASON == "deepseek-v4-pro".
ADVERSARIAL CLASSES: malformed_input — bad ASSET_UNIVERSE string must parse to list or raise clearly; dirty_worktree — .env must stay gitignored.
ARTIFACT: config.py + .env.example + test results.
```

### W1-T3 — Data ingestion + indicator feature store
```
TASK: Implement trading/data/ingest.py and trading/data/features.py.
DELIVERABLE:
  - ingest.py: fetch_historical(symbol, start, end) via yfinance returning OHLCV DataFrame; fetch_realtime(symbol) via alpaca-py MarketDataClient (skip if no keys); cache to trading/state/<symbol>.parquet or sqlite.
  - features.py: add indicators with pandas-ta: sma/ema(20,50,200), rsi(14), macd, bbands, atr(14), volume z-score. Return enriched DataFrame.
SCOPE: only trading/data/.
BASELINE TEST: tests/test_data.py::test_empty — calling features on an empty frame returns frame unchanged (no crash).
FAILING-FIRST PROOF: tests/test_data.py::test_fetch_and_features — with yfinance (network) OR a synthetic fixture, asserts RSI_14 column exists and is finite after fill.
IMPLEMENTATION CONSTRAINTS: retry/backoff on network; never log API keys; make Alpaca calls no-op when keys absent.
AUTOMATED VERIFICATION: `pytest tests/test_data.py -q` (skip network test if offline via @pytest.mark.skip).
MANUAL QA: `python -c "from trading.data.features import add_features; import pandas as pd; df=pd.DataFrame(...); print(add_features(df).columns)"` shows indicator columns.
ADVERSARIAL CLASSES: hung_commands — network calls wrapped with timeout; stale_state — cache must invalidate on new date range; misleading_success — assert column NON-NaN, not just presence.
ARTIFACT: ingest.py + features.py + test log.
```
