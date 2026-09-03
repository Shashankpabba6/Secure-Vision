# trading-bot-build - Work Plan

## TL;DR (For humans)
**What you'll get:** A self-hosted, 24/7 trading bot as a Python package inside this workspace. Two AI agents — a Chart Analysis agent that reads price history and detects patterns, and a Decision agent that decides buy/sell/hold with position sizing — coordinated by a LangGraph orchestrator, guarded by a Risk agent, and watched by a 24/7 Supervisor agent that alerts you on Telegram. It runs on paper trading first (Alpaca), with one flag to go live.

**Why this approach:** TradingAgents (built on LangGraph) gives you the two-agent design plus a backtest harness out of the box; LangGraph's checkpointing is what makes 24/7 crash-resume safe; DeepSeek V4 (Flash/Pro) is the cheapest frontier-class reasoning and MiniMax M3 is the cheapest vision for chart images.

**What it will NOT do:** Trade with real money unless you flip `MODE=live` (paper by default); guarantee profits; do high-frequency trading (LLM agents are too slow for that); trade crypto unless you later add the Binance adapter.

**Effort:** Large
**Risk:** Medium - financial/API-key risk contained by paper-first + Risk agent kill-switch.

**Decisions to sanity-check (adopted defaults - veto any at the gate):**
- Execution: paper trading first, `MODE=live` behind a flag
- Asset class: US stocks via Alpaca
- Framework: TradingAgents (LangGraph-based) + custom Supervisor agent
- Models: DeepSeek V4 Flash (routine) + V4 Pro (reasoning) for text; MiniMax M3 for chart vision
- Alerts: Telegram

Your next move: approve, or run a high-accuracy review first. Full execution detail follows below.

> TL;DR (machine): Large effort, Medium risk; two-agent LangGraph trading bot, paper-first Alpaca, DeepSeek V4 + MiniMax M3, 24/7 supervisor, backtest harness.

---

## Scope

### Must have
- `trading/` Python package inside this repo (WebApp/)
- Data ingestion (historical + real-time OHLCV) + indicator feature store
- Chart Analysis Agent (Agent B): quant pattern detection + optional vision confirmation -> scored setup JSON
- Decision Agent (Agent A): BUY/SELL/HOLD + size_fraction + entry/stop/target + confidence
- Risk Agent: position sizing, max-drawdown cap, kill-switch
- Broker adapter: Alpaca paper/live, abstracted interface (Binance pluggable later)
- LangGraph orchestration with SQLite/Postgres checkpointing (24/7 crash-resume)
- 24/7 Supervisor/Monitoring agent + Telegram alerts + Docker/systemd + heartbeat
- Backtesting harness (TradingAgents backtest or backtrader) with Sharpe/drawdown/hit-rate
- Optional Streamlit dashboard reusing the existing Streamlit stack

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Live trading enabled by default (paper only)
- HFT / sub-second execution
- Crypto/Binance execution (interface left pluggable, not implemented)
- Any guaranteed-return claim or financial advice in code/docs
- Secrets stored in code (must use `.env` / secret manager)

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + tests-after; framework `pytest`
- Evidence: `.omo/evidence/trading-bot-build/task-<N>.<ext>` (outside ulw-loop use `.omo/evidence/`)

## Execution strategy

### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Wave 1 (1-3): project foundation
- Wave 2 (4-7): agents (chart, decision, risk, supervisor)
- Wave 3 (8-10): broker, orchestration, process supervisor
- Wave 4 (11-12): backtest + observability
- Wave 5 (13): optional UI

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | - | 2,3 | - |
| 2 | 1 | 4,5,6,7,8,9 | 3 |
| 3 | 1,2 | 4,5 | - |
| 4 | 2,3 | 9 | 5,6,7 |
| 5 | 2,3,4 | 9 | 6,7 |
| 6 | 2 | 5,9 | 7 |
| 7 | 2 | 9 | 4,5,6 |
| 8 | 2 | 9 | 3,4 |
| 9 | 4,5,6,7,8 | 10,11 | - |
| 10 | 9 | 11 | - |
| 11 | 9,10 | 12 | - |
| 12 | 11 | 13 | - |
| 13 | 9,12 | - | - |

## Todos

- [ ] 1. Scaffold `trading/` package and dependency manifest
  What to do / Must NOT do: Create `trading/` with `pyproject.toml`/`requirements.txt` listing `langgraph`, `langchain`, `tradingagents`, `alpaca-py`, `pandas`, `pandas-ta`, `ta` (TA-Lib wrapper), `yfinance`, `python-dotenv`, `pydantic-settings`, `apscheduler`, `python-telegram-bot`, `backtrader`, `streamlit`, `matplotlib`, `mplfinance`. Create subpackages `trading/{config,data,agents,broker,orchestration,monitor,backtest,ui}`. Must NOT commit a `.env` or hardcode versions that conflict with the existing repo.
  Parallelization: Wave 1 | Blocked by: - | Blocks: 2,3
  References (executor has NO interview context - be exhaustive): langchain-ai.github.io/langgraph ; TradingAgents-AI/TradingAgents README (github.com/TradingAgents-AI/TradingAgents) ; existing repo is Python/Streamlit (WebApp/final_app.py).
  Acceptance criteria (agent-executable): `pip install -r requirements.txt` succeeds AND `python -c "import trading"` succeeds AND `pytest --collect-only -q` discovers the package.
  QA scenarios (name the exact tool + invocation): happy - `pytest --collect-only` lists 0 collected errors; failure - a missing dep surfaces as a clear pip error, not a silent import. Evidence `.omo/evidence/trading-bot-build/task-1.env`.
  Commit: Y | feat(trading): scaffold package and dependency manifest

- [ ] 2. Typed config & secrets module (paper/live mode + model selection)
  What to do / Must NOT do: `trading/config.py` using `pydantic-settings` loads `.env`: `MODE` (paper|live, default paper), `ASSET_UNIVERSE` (watchlist file or comma list), `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL=https://api.deepseek.com`, `MINIMAX_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL` (paper: `https://paper-api.alpaca.markets`), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `MODEL_QUICK=deepseek-v4-flash`, `MODEL_REASON=deepseek-v4-pro`, `MODEL_VISION=minimax-m3`, `KILL_SWITCH=0`. Validate on import. Must NOT hardcode keys or commit `.env`.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 4,5,6,7,8,9
  References: api-docs.deepseek.com/quick_start/pricing (model ids `deepseek-v4-flash`/`deepseek-v4-pro`, base url) ; Alpaca paper trading docs (alpaca.markets) ; pydantic-settings docs.
  Acceptance criteria: `python -c "from trading.config import settings; assert settings.MODE in ('paper','live')"` passes; running with a missing required key raises a clear `ValidationError`.
  QA scenarios: happy - loads from `.env` and exposes `settings.MODEL_REASON`; failure - delete `DEEPSEEK_API_KEY` -> import raises ValidationError naming the missing field. Evidence `.omo/evidence/trading-bot-build/task-2.env`.
  Commit: Y | feat(config): typed settings with paper/live mode and model selection

- [ ] 3. Data ingestion + indicator feature store
  What to do / Must NOT do: `trading/data/ingest.py` fetches historical OHLCV via `yfinance` (backtest) and real-time via Alpaca `StockBarsRequest` (paper/live) for `ASSET_UNIVERSE`; cache to SQLite/parquet. `trading/data/features.py` computes indicators with `pandas-ta`/`ta`: SMA/EMA(20,50,200), RSI(14), MACD, BBANDS, ATR(14), volume Z-score. Must NOT block on rate limits (add retry/backoff); never log API keys.
  Parallelization: Wave 1 | Blocked by: 1,2 | Blocks: 4,5
  References: yfinance docs (pypi.org/project/yfinance) ; alpaca-py `StockBarsRequest` ; pandas-ta readme (github.com/twopirllc/pandas-ta) ; TA-Lib.
  Acceptance criteria: `pytest tests/test_data.py::test_fetch_and_features` fetches one ticker and asserts an `RSI_14` column exists with finite, non-NaN-after-fill values.
  QA scenarios: happy - returns a DataFrame with indicator columns; failure - simulate network error -> typed `DataError` after retries, no infinite loop. Evidence `.omo/evidence/trading-bot-build/task-3.parquet`.
  Commit: Y | feat(data): historical+realtime OHLCV ingestion and indicator features

- [ ] 4. Chart Analysis Agent (Agent B) - patterns + vision confirmation
  What to do / Must NOT do: `trading/agents/chart_agent.py` - input: features DataFrame + ticker + timeframe. Step 1: quant pattern detection (head & shoulders, double top/bottom, triangles, flags, RSI divergence) -> candidate patterns with confidence. Step 2 (optional, config-gated): render candlestick via `mplfinance` to PNG, call MiniMax M3 (OpenAI-compatible) to confirm pattern + support/resistance. Output Pydantic `ChartSetup {patterns:list, support, resistance, trend, attractiveness_score:0-100, vision_confirmed:bool, rationale}`. Must NOT emit trades - analysis only.
  Parallelization: Wave 2 | Blocked by: 2,3 | Blocks: 9
  References: MiniMax M3 vision pricing ($0.60/$2.40, native image) - andrew.ooo/minimax-m3 ; mplfinance docs ; pandas-ta pattern helpers ; DeepSeek is text-only so vision uses MiniMax (per prior research).
  Acceptance criteria: `pytest tests/test_chart_agent.py` returns a `ChartSetup` with a non-empty `patterns` list and a 0-100 `attractiveness_score` for a fixture ticker; vision call mocked.
  QA scenarios: happy - detects a head & shoulders on a synthetic series; failure - vision API down -> falls back to quant-only with `vision_confirmed=False`, no crash. Evidence `.omo/evidence/trading-bot-build/task-4.json`.
  Commit: Y | feat(agent): chart analysis agent with quant patterns + optional vision confirmation

- [ ] 5. Decision Agent (Agent A) - BUY/SELL/HOLD + sizing via DeepSeek V4
  What to do / Must NOT do: `trading/agents/decision_agent.py` - LangGraph/TradingAgents node consuming `ChartSetup` + market state + portfolio. Uses DeepSeek V4 Pro (reasoning) for the final call, V4 Flash for routine screening, via OpenAI-compatible client at DeepSeek base url with structured JSON output. Output Pydantic `TradeDecision {signal:BUY|SELL|HOLD, size_fraction, entry_price, stop_loss, target_price, horizon_days, confidence, rationale}`. Must NOT place orders - hand off to Risk agent.
  Parallelization: Wave 2 | Blocked by: 2,3,4 | Blocks: 9
  References: api-docs.deepseek.com (deepseek-v4-pro/deepseek-v4-flash, JSON mode, tool calls, $0.435/$0.87 and $0.14/$0.28 per 1M) ; TradingAgents `TradeRecommendation` schema (github.com/TradingAgents-AI/TradingAgents).
  Acceptance criteria: `pytest tests/test_decision_agent.py` with a fixture `ChartSetup` returns a valid `TradeDecision` with `signal in {BUY,SELL,HOLD}` and `0 <= size_fraction <= 1`.
  QA scenarios: happy - bullish setup -> BUY with `size_fraction=0.3`; failure - LLM returns malformed JSON -> parser retries once then defaults to conservative HOLD. Evidence `.omo/evidence/trading-bot-build/task-5.json`.
  Commit: Y | feat(agent): decision agent emitting structured trade decisions via DeepSeek V4

- [ ] 6. Risk Agent - sizing caps, drawdown halt, kill-switch
  What to do / Must NOT do: `trading/agents/risk_agent.py` - input `TradeDecision` + portfolio + drawdown state. Enforce: max position fraction (e.g. 0.25), per-trade risk <= 2% of equity (size from ATR-based stop), max portfolio drawdown cap (e.g. 10% -> flatten & halt), `KILL_SWITCH=1` forces HOLD/flatten. Output sanitized order intent. Must NOT invent sizing beyond caps or bypass kill-switch.
  Parallelization: Wave 2 | Blocked by: 2 | Blocks: 5,9
  References: standard position-sizing / drawdown-halt risk management (per prior research: risk agent is mandatory for LLM trading bots).
  Acceptance criteria: `pytest tests/test_risk.py` - oversized decision truncated to cap; drawdown over cap sets `halt=True`; `KILL_SWITCH=1` forces HOLD.
  QA scenarios: happy - BUY size 0.9 truncated to 0.25; failure - kill-switch mid-run -> zero orders emitted. Evidence `.omo/evidence/trading-bot-build/task-6.json`.
  Commit: Y | feat(agent): risk agent with sizing caps, drawdown halt, kill-switch

- [ ] 7. Monitoring / Supervisor Agent (24/7) - health checks + Telegram alerts
  What to do / Must NOT do: `trading/monitor/supervisor.py` - APScheduler job every N min checks: (a) decision-loop heartbeat fresh, (b) open positions at broker match expected plan, (c) Alpaca/DeepSeek/MiniMax API connectivity, (d) exception count in window, (e) process alive. On anomaly -> `python-telegram-bot` message to `TELEGRAM_CHAT_ID`. Expose `/health`. Rule-based + optional cheap LLM (V4 Flash) summary. Must NOT place trades - observe + alert only.
  Parallelization: Wave 2 | Blocked by: 2 | Blocks: 9
  References: APScheduler docs (apscheduler.readthedocs.io) ; python-telegram-bot docs ; Healthchecks.io (healthchecks.io) for external heartbeat.
  Acceptance criteria: `pytest tests/test_supervisor.py` with a fake clock + mocked broker asserts Telegram send is called when heartbeat is stale and NOT called when healthy.
  QA scenarios: happy - simulated stale heartbeat -> alert sent; failure - invalid Telegram token -> logs error, loop continues. Evidence `.omo/evidence/trading-bot-build/task-7.log`.
  Commit: Y | feat(monitor): 24/7 supervisor agent with health checks and Telegram alerts

- [ ] 8. Broker adapter (Alpaca) with pluggable interface
  What to do / Must NOT do: `trading/broker/base.py` abstract `Broker` interface; `trading/broker/alpaca.py` wraps `alpaca-py` `TradingClient` + `MarketOrderRequest`: `get_positions()`, `submit_order(symbol, qty, side, type)`, `get_account()`, `close_all()`. Use paper base url when `MODE=paper`. Must NOT submit real orders when `MODE!=live`; must respect Risk-agent caps.
  Parallelization: Wave 3 | Blocked by: 2 | Blocks: 9
  References: alpaca-py docs (TradingClient, MarketOrderRequest, paper API at alpaca.markets/docs) ; existing repo is Python so `alpaca-py` fits.
  Acceptance criteria: `pytest tests/test_broker.py` with `AlpacaClient` mocked asserts `submit_order` builds the correct request; integration test against Alpaca paper sandbox skipped if keys absent.
  QA scenarios: happy - market buy 10 AAPL builds a valid order; failure - insufficient buying power -> raises typed error, no partial-fill hack. Evidence `.omo/evidence/trading-bot-build/task-8.json`.
  Commit: Y | feat(broker): Alpaca adapter with paper/live mode and pluggable interface

- [ ] 9. LangGraph orchestration with checkpointing (24/7 crash-resume)
  What to do / Must NOT do: `trading/orchestration/graph.py` - `StateGraph` nodes chart_agent -> decision_agent -> risk_agent -> broker_adapter, conditional edges (HOLD skips broker). Persistent `SqliteSaver` (or Postgres) checkpointer so state survives crashes. `trading/main.py` - entrypoint: load config, build graph, loop on market-hours schedule invoking `propagate()`; on exception the graph resumes from last checkpoint. Must NOT lose state on restart or run outside market hours.
  Parallelization: Wave 3 | Blocked by: 4,5,6,7,8 | Blocks: 10,11
  References: langchain-ai.github.io/langgraph (checkpointer, persistence, time-travel) ; TradingAgents graph design (github.com/TradingAgents-AI/TradingAgents) - this IS the 24/7 resilience core.
  Acceptance criteria: `pytest tests/test_orchestration.py` runs a full cycle producing a `TradeDecision`; a forced restart mid-graph resumes and does NOT duplicate an order (checkpoint/idempotency test).
  QA scenarios: happy - end-to-end paper cycle emits exactly one order; failure - crash after risk but before broker -> on resume broker step runs once (no double fill). Evidence `.omo/evidence/trading-bot-build/task-9.db`.
  Commit: Y | feat(orchestration): LangGraph state graph with crash-resume checkpointer

- [ ] 10. Process supervisor + heartbeat (Docker/systemd)
  What to do / Must NOT do: `Dockerfile` (python:3.12-slim) + `docker-compose.yml` (`restart: unless-stopped`, `env_file:.env`, volume for state) ; optional `trading-bot.service` systemd unit (`Restart=always`). Add heartbeat ping to Healthchecks.io each loop; missed ping alerts you. Document in README. Must NOT run as root; must not leak secrets (`.dockerignore` excludes `.env`).
  Parallelization: Wave 3 | Blocked by: 9 | Blocks: 11
  References: Docker best practices (non-root user, env_file) ; Healthchecks.io docs ; systemd unit docs (`Restart=always`).
  Acceptance criteria: `docker build -t trading-bot .` succeeds; `docker run` starts and prints a heartbeat; `systemd-analyze verify trading-bot.service` passes.
  QA scenarios: happy - `restart:unless-stopped` recovers after `docker kill`; failure - missing `.env` -> container exits with clear error. Evidence `.omo/evidence/trading-bot-build/task-10.log`.
  Commit: Y | feat(ops): Docker/systemd supervisor with healthcheck heartbeat for 24/7

- [ ] 11. Backtesting harness with performance report
  What to do / Must NOT do: `trading/backtest/run.py` - drive the same agents over yfinance history (>=1yr) for `ASSET_UNIVERSE`; support `--dry-run` (stub LLM) and `--budget-cap-usd` (cost tracking). Produce report: Sharpe, max drawdown, hit-rate, expectancy, vs buy&hold. Must NOT use future data (no look-ahead); must NOT trade live.
  Parallelization: Wave 4 | Blocked by: 9,10 | Blocks: 12
  References: TradingAgents backtest harness (`tradingagents backtest`, `--dry-run`, `--budget-cap-usd`) ; backtrader docs (backtrader.com) ; yfinance.
  Acceptance criteria: `pytest tests/test_backtest.py` runs a dry-run backtest on cached data and asserts the report contains Sharpe and drawdown; `python -m trading.backtest --dry-run` exits 0 with report printed.
  QA scenarios: happy - dry-run over AAPL history yields metrics; failure - insufficient history -> clear error, no crash. Evidence `.omo/evidence/trading-bot-build/task-11.report`.
  Commit: Y | feat(backtest): historical backtest harness with performance report

- [ ] 12. Observability & evaluation loop
  What to do / Must NOT do: `trading/observability.py` - structured logging of every decision (inputs/outputs, token cost via DeepSeek cache rates), metrics (P&L, win rate, drawdown) persisted; optional LangSmith tracing for LangGraph runs; weekly eval that replays past decisions vs realized returns. Token cost tracker. Must NOT log secrets; must not exceed budget silently.
  Parallelization: Wave 4 | Blocked by: 11 | Blocks: 13
  References: LangSmith tracing docs ; DeepSeek pricing (cache-hit $0.003625/$0.0028) for cost tracking.
  Acceptance criteria: `pytest tests/test_observability.py` asserts a decision is logged with redacted keys and cost recorded; `python -m trading.eval` runs a replay.
  QA scenarios: happy - log line contains no API-key substring; failure - cost exceeds budget cap -> warning raised. Evidence `.omo/evidence/trading-bot-build/task-12.log`.
  Commit: Y | feat(observability): decision logging, metrics, and cost tracking

- [ ] 13. Streamlit dashboard (optional, read-only)
  What to do / Must NOT do: `trading/ui/app.py` - reuse existing Streamlit pattern (WebApp/final_app.py) to show positions, today's signals, P&L, agent logs, and a manual KILL-SWITCH button. Runs as a separate `streamlit run` process. Must NOT place orders; read-only + kill-switch only.
  Parallelization: Wave 5 | Blocked by: 9,12 | Blocks: -
  References: WebApp/final_app.py (existing Streamlit structure) ; streamlit docs.
  Acceptance criteria: `streamlit run trading/ui/app.py` boots; page shows positions from a broker mock; kill-switch button sets `KILL_SWITCH`.
  QA scenarios: happy - dashboard renders with sample data; failure - broker down -> shows error state, no crash. Evidence `.omo/evidence/trading-bot-build/task-13.png`.
  Commit: Y | feat(ui): read-only Streamlit dashboard with kill-switch

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
- [ ] F2. Code quality review
- [ ] F3. Real manual QA
- [ ] F4. Scope fidelity

## Commit strategy
- One conventional commit per todo (`feat(trading): ...`, `feat(config): ...`, `feat(agent): ...`, `feat(broker): ...`, `feat(orchestration): ...`, `feat(ops): ...`, `feat(backtest): ...`, `feat(observability): ...`, `feat(ui): ...`).
- Branch `feat/trading-bot`; open PR only after all todos green and the final verification wave passes.
- `.env`, `.omo/`, `__pycache__`, model checkpoints in `.gitignore`.

## Success criteria
- Bot runs 24/7 on paper and resumes after a forced crash (checkpoint verified by task 9 test).
- One full decision cycle (B -> A -> Risk -> Broker) completes end-to-end on paper without manual intervention.
- Backtest over >=1yr history produces a Sharpe / max-drawdown / hit-rate report (task 11).
- Supervisor alerts fire on a simulated failure (stale heartbeat, API down) (task 7).
- Zero secrets in code; all via `.env` (task 2).
- `docker build` + `systemd-analyze verify` pass (task 10).
