# Worker Specs — Wave 2 (Agents)

> Companion to `worker-specs-1.md`. Paste each block as a single prompt to an implementation agent.
> Depends on Wave 1 (T1-T3) being complete.

---

### W2-T4 — Chart Analysis Agent (Agent B)
```
TASK: Implement trading/agents/chart_agent.py.
DELIVERABLE: function/class `analyze_chart(features_df, ticker, timeframe) -> ChartSetup` where ChartSetup is a pydantic model:
  patterns: list[str], support: float|None, resistance: float|None, trend: str,
  attractiveness_score: int (0-100), vision_confirmed: bool, rationale: str
  Step 1: quant pattern scan (head&shoulders, double top/bottom, triangles, flags, RSI divergence) with confidence.
  Step 2 (config-gated, default OFF): render candlestick via mplfinance to PNG, call MiniMax M3 (OpenAI-compatible) for confirmation;
    on failure set vision_confirmed=False and fall back to quant-only.
SCOPE: only trading/agents/chart_agent.py + its pydantic model (put model in trading/agents/models.py).
BASELINE TEST: tests/test_chart_agent.py::test_model_valid — ChartSetup with valid fields constructs.
FAILING-FIRST PROOF: tests/test_chart_agent.py::test_detects_pattern — on a synthetic head&shoulders OHLCV, returns patterns containing "head_and_shoulders" and score in 0-100.
IMPLEMENTATION CONSTRAINTS: must NOT emit trades; no hardcoded thresholds without config; vision call mocked in tests.
AUTOMATED VERIFICATION: `pytest tests/test_chart_agent.py -q`.
MANUAL QA: `python -c "from trading.agents.chart_agent import analyze_chart; ..."` on a sample DataFrame prints a ChartSetup JSON.
ADVERSARIAL CLASSES: malformed_input — garbage DataFrame raises typed error; misleading_success — score must be computed, not hardcoded; stale_state — vision PNG temp file deleted after call.
ARTIFACT: chart_agent.py + models.py + test log.
```

### W2-T5 — Decision Agent (Agent A)
```
TASK: Implement trading/agents/decision_agent.py.
DELIVERABLE: `decide(chart_setup, market_state, portfolio) -> TradeDecision` pydantic model:
  signal: Literal["BUY","SELL","HOLD"], size_fraction: float (0-1), entry_price, stop_loss,
  target_price, horizon_days: int|None, confidence: float (0-1), rationale: str
  Uses DeepSeek V4 Pro (reasoning) via OpenAI-compatible client at DEEPSEEK_BASE_URL with JSON/function-call output;
  routine screening may use V4 Flash. Build the client from trading.config settings.
SCOPE: only trading/agents/decision_agent.py (+ models in trading/agents/models.py).
BASELINE TEST: tests/test_decision_agent.py::test_model_valid — TradeDecision with valid fields constructs and signal in set.
FAILING-FIRST PROOF: tests/test_decision_agent.py::test_decision_shape — with a fixture ChartSetup, mocked LLM returns a parsed TradeDecision with 0<=size_fraction<=1.
IMPLEMENTATION CONSTRAINTS: must NOT place orders; hand off to Risk agent; parser must retry once then default to conservative HOLD on malformed JSON.
AUTOMATED VERIFICATION: `pytest tests/test_decision_agent.py -q` (LLM mocked).
MANUAL QA: with DEEPSEEK_API_KEY set, run a single decision on a sample setup and print the TradeDecision.
ADVERSARIAL CLASSES: malformed_input — malformed LLM JSON -> HOLD fallback; prompt_injection — untrusted text in chart_setup.rationale must not alter tool schema; misleading_success — assert parsed fields, not just non-empty response.
ARTIFACT: decision_agent.py + test log.
```

### W2-T6 — Risk Agent
```
TASK: Implement trading/agents/risk_agent.py.
DELIVERABLE: `apply_risk(decision: TradeDecision, portfolio, drawdown_state, settings) -> OrderIntent | Halt`
  Enforce: max position fraction (0.25), per-trade risk <= 2% equity (size from ATR stop),
  max portfolio drawdown cap (0.10) -> set halt=True and flatten, KILL_SWITCH==1 -> force HOLD/flatten.
  Output a sanitized order intent (symbol, qty, side) or a Halt signal.
SCOPE: only trading/agents/risk_agent.py.
BASELINE TEST: tests/test_risk.py::test_pass_through — a small valid decision returns an OrderIntent unchanged.
FAILING-FIRST PROOF: tests/test_risk.py::test_caps — oversized BUY (size 0.9) truncated to 0.25; drawdown>cap sets halt; KILL_SWITCH=1 forces HOLD.
IMPLEMENTATION CONSTRAINTS: must NOT invent sizing beyond caps; must not bypass kill-switch.
AUTOMATED VERIFICATION: `pytest tests/test_risk.py -q`.
MANUAL QA: feed an oversized decision, assert returned qty respects cap.
ADVERSARIAL CLASSES: repeated_interruptions — kill-switch evaluated on every call; misleading_success — assert halt flag is the actual control flow, not just logged.
ARTIFACT: risk_agent.py + test log.
```

### W2-T7 — Monitoring / Supervisor Agent (24/7)
```
TASK: Implement trading/monitor/supervisor.py.
DELIVERABLE: APScheduler job (every N min, configurable) checking: (a) decision-loop heartbeat fresh,
  (b) open positions at broker match expected plan, (c) Alpaca/DeepSeek/MiniMax API connectivity,
  (d) exception count in window, (e) process alive. On anomaly -> python-telegram-bot message to TELEGRAM_CHAT_ID.
  Expose `health()` returning status dict. Rule-based; optional cheap LLM (V4 Flash) for alert text.
SCOPE: only trading/monitor/supervisor.py.
BASELINE TEST: tests/test_supervisor.py::test_health_ok — health() returns dict with all keys when healthy.
FAILING-FIRST PROOF: tests/test_supervisor.py::test_alert_on_stale — with fake clock + mocked broker + mocked telegram, stale heartbeat triggers exactly one send.
IMPLEMENTATION CONSTRAINTS: must NOT place trades; observe + alert only; telegram token invalid -> log error, loop continues.
AUTOMATED VERIFICATION: `pytest tests/test_supervisor.py -q`.
MANUAL QA: run supervisor with a forced stale heartbeat, assert Telegram send (mock) called once.
ADVERSARIAL CLASSES: hung_commands — each check has timeout; misleading_success — assert send was actually invoked, not just "would send" logged; cancel_resume — job reschedules cleanly.
ARTIFACT: supervisor.py + test log.
```
