# Worker Specs — Wave 3 (Orchestration & Execution)

> Companion to `worker-specs-1.md` / `worker-specs-2.md`. Paste each block as a single prompt.
> Depends on Wave 1 and Wave 2 being complete.

---

### W3-T8 — Broker adapter (Alpaca)
```
TASK: Implement trading/broker/base.py (abstract Broker) and trading/broker/alpaca.py.
DELIVERABLE: Broker interface: get_positions(), submit_order(symbol, qty, side, type="market"),
  get_account(), close_all(). Alpaca impl wraps alpaca-py TradingClient + MarketOrderRequest; use paper base url when MODE=="paper".
SCOPE: only trading/broker/.
BASELINE TEST: tests/test_broker.py::test_interface — a FakeBroker implementing base passes isinstance check.
FAILING-FIRST PROOF: tests/test_broker.py::test_submit_builds_request — mocked AlpacaClient.submit_order builds correct MarketOrderRequest; integration test vs Alpaca paper skipped if no keys.
IMPLEMENTATION CONSTRAINTS: must NOT submit real orders when MODE!="live"; must respect Risk caps (assert qty>0).
AUTOMATED VERIFICATION: `pytest tests/test_broker.py -q`.
MANUAL QA: with paper keys, submit a 1-share market buy of a watchlist symbol in paper, then close_all(); verify no real fill risk.
ADVERSARIAL CLASSES: malformed_input — qty<=0 rejected; dirty_worktree — no change to existing files; misleading_success — assert order id returned, not just "submitted" string.
ARTIFACT: alpaca.py + base.py + test log.
```

### W3-T9 — LangGraph orchestration with checkpointing (24/7 crash-resume)
```
TASK: Implement trading/orchestration/graph.py and trading/main.py.
DELIVERABLE: StateGraph nodes chart_agent -> decision_agent -> risk_agent -> broker_adapter with conditional edges
  (HOLD skips broker). Persistent SqliteSaver checkpointer (trading/state/graph_checkpoints.db) so state survives crashes.
  main.py: load config, build graph, loop on market-hours schedule invoking propagate(); on exception resume from last checkpoint.
SCOPE: only trading/orchestration/ and trading/main.py.
BASELINE TEST: tests/test_orchestration.py::test_graph_builds — graph compiles with all 4 nodes.
FAILING-FIRST PROOF: tests/test_orchestration.py::test_full_cycle — a full cycle produces a TradeDecision (broker mocked);
  tests/test_orchestration.py::test_crash_resume — force restart mid-graph, assert broker step runs exactly once (no double fill) via checkpoint idempotency.
IMPLEMENTATION CONSTRAINTS: must NOT lose state on restart; must NOT run outside market hours (configurable schedule).
AUTOMATED VERIFICATION: `pytest tests/test_orchestration.py -q`.
MANUAL QA: run main.py against paper with a forced crash after risk, confirm on resume order placed once (check broker mock call count == 1).
ADVERSARIAL CLASSES: cancel_resume — checkpoint resume is the core proof; stale_state — old checkpoint from a prior day must not replay; repeated_interruptions — multiple crashes don't multiply orders.
ARTIFACT: graph.py + main.py + test log + checkpoint db path.
```

### W3-T10 — Process supervisor + heartbeat (Docker/systemd)
```
TASK: Add ops files for 24/7.
DELIVERABLE: Dockerfile (python:3.12-slim, non-root user), docker-compose.yml (restart: unless-stopped,
  env_file:.env, volume for trading/state), optional trading-bot.service systemd unit (Restart=always),
  and a heartbeat ping to Healthchecks.io each loop in main.py (or supervisor). README section documenting deploy.
SCOPE: new files only (Dockerfile, docker-compose.yml, trading-bot.service, README addition). No edits to existing app code.
BASELINE TEST: `docker compose config` validates compose; `systemd-analyze verify trading-bot.service` passes (if systemd present).
FAILING-FIRST PROOF: `docker build -t trading-bot .` succeeds and `docker run --rm trading-bot python -c "import trading"` exits 0.
IMPLEMENTATION CONSTRAINTS: non-root; .dockerignore excludes .env; heartbeat ping skipped gracefully if no Healthchecks URL.
AUTOMATED VERIFICATION: `docker build -t trading-bot .` (or note if Docker absent), `systemd-analyze verify trading-bot.service`.
MANUAL QA: `docker run --rm trading-bot` boots and prints a heartbeat line; `docker kill` then `docker run` recovers with restart:unless-stopped.
ADVERSARIAL CLASSES: dirty_worktree — .env must not be copied into image (verify with `docker run --rm trading-bot cat .env` fails); hung_commands — build has no interactive prompts.
ARTIFACT: Dockerfile + compose + service + README + build/log verify output.
```
