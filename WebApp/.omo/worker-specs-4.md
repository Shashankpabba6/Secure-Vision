# Worker Specs — Wave 4, Wave 5 & Final Verification

> Companion to `worker-specs-1/2/3.md`. Paste each block as a single prompt.
> Wave 4 depends on Wave 3; Wave 5 depends on Wave 3 + 4.

---

### W4-T11 — Backtesting harness
```
TASK: Implement trading/backtest/run.py.
DELIVERABLE: drive the SAME agents over yfinance history (>=1yr) for ASSET_UNIVERSE. Support --dry-run (stub LLM)
  and --budget-cap-usd (cost tracking). Output report: Sharpe, max drawdown, hit-rate, expectancy, vs buy&hold.
  Reuse TradingAgents backtest if available, else a backtrader loop calling the agents.
SCOPE: only trading/backtest/.
BASELINE TEST: tests/test_backtest.py::test_report_shape — report dict has sharpe & max_drawdown keys.
FAILING-FIRST PROOF: tests/test_backtest.py::test_dry_run — `python -m trading.backtest --dry-run` exits 0 and prints a report on cached/sample data.
IMPLEMENTATION CONSTRAINTS: NO future/look-ahead data; NO live trading.
AUTOMATED VERIFICATION: `pytest tests/test_backtest.py -q`.
MANUAL QA: `python -m trading.backtest --dry-run` prints a metrics table.
ADVERSARIAL CLASSES: misleading_success — metrics must be computed from real backtest, not constants; stale_state — use cached OHLCV, not regenerated; hung_commands — bound runtime.
ARTIFACT: run.py + test log + sample report.
```

### W4-T12 — Observability & evaluation loop
```
TASK: Implement trading/observability.py.
DELIVERABLE: structured logging of every decision (inputs/outputs, token cost via DeepSeek cache rates), metrics (P&L, win rate, drawdown)
  persisted; optional LangSmith tracing for LangGraph runs; weekly eval that replays past decisions vs realized returns. Token cost tracker.
SCOPE: only trading/observability.py + a replay script trading/eval.py.
BASELINE TEST: tests/test_observability.py::test_log_shape — a logged decision has expected keys.
FAILING-FIRST PROOF: tests/test_observability.py::test_no_secret_leak — log line for a decision contains no API-key substring; cost recorded.
IMPLEMENTATION CONSTRAINTS: must NOT log secrets; must not exceed budget silently (warn on cap).
AUTOMATED VERIFICATION: `pytest tests/test_observability.py -q`.
MANUAL QA: run one decision cycle, assert log file has no key substring and a cost field.
ADVERSARIAL CLASSES: misleading_success — assert redaction actually applied (grep the log); hung_commands — eval replay bounded.
ARTIFACT: observability.py + eval.py + sample log.
```

### W5-T13 — Streamlit dashboard (optional, read-only)
```
TASK: Implement trading/ui/app.py.
DELIVERABLE: reuse existing Streamlit pattern (WebApp/final_app.py) to show positions, today's signals, P&L, agent logs,
  and a manual KILL-SWITCH button. Runs as a separate `streamlit run` process.
SCOPE: only trading/ui/app.py.
BASELINE TEST: tests/test_ui.py::test_import — `import trading.ui.app` succeeds.
FAILING-FIRST PROOF: tests/test_ui.py::test_kill_switch — clicking kill-switch (simulated) sets KILL_SWITCH env/flag.
IMPLEMENTATION CONSTRAINTS: must NOT place orders; read-only + kill-switch only.
AUTOMATED VERIFICATION: `pytest tests/test_ui.py -q`; `streamlit run trading/ui/app.py --server.headless` boots without error.
MANUAL QA: boot dashboard with broker mock, confirm positions panel renders.
ADVERSARIAL CLASSES: dirty_worktree — no edits to final_app.py; misleading_success — assert panel shows data, not placeholder.
ARTIFACT: app.py + test log.
```

---

## Final Verification Wave (run after ALL todos)

Paste these as separate review prompts after the build completes:

### F1. Plan compliance audit
```
TASK: Audit the implemented trading/ package against .omo/plans/trading-bot-build.md.
Verify every todo (1-13) is implemented, every acceptance criterion met, and no Must-NOT-have was violated
(paper-only by default, no HFT, no secrets in code, no crypto execution). Report a checklist with PASS/FAIL per todo.
```

### F2. Code quality review
```
TASK: Review trading/ for code quality: typing, no hardcoded secrets, error handling, docstrings, test coverage.
Flag anything that would fail in production (e.g., unbounded retries, unhandled API errors). Return a prioritized list.
```

### F3. Real manual QA
```
TASK: Spin up the bot on PAPER mode with stubbed LLMs (--dry-run backtest + a mocked decision_agent).
Prove: (a) one full cycle B->A->Risk->Broker runs end-to-end on paper, (b) forcing a crash after risk and resuming
places the order exactly once (checkpoint idempotency), (c) supervisor alert fires on a simulated stale heartbeat.
Capture screenshots/logs as evidence.
```

### F4. Scope fidelity
```
TASK: Confirm the deliverable matches the plan's Scope IN (all 10 items) and Scope OUT (the 5 guardrails).
Explicitly confirm: live trading is OFF by default, crypto is not executed, no financial-advice claims in code/docs.
```
