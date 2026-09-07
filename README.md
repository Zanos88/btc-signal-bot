# btc-signal-bot — Stage 2 (Signal + dry-run-gated execution)

BTC-PERP multi-timeframe confluence signal bot for a Propr.xyz 1-Step
Classic challenge account. It posts Telegram alerts and (Stage 2) can
drive a Propr execution layer — but **order dispatch is OFF by default,
behind a two-switch safety gate**: a real order is sent only when BOTH
`DRY_RUN=false` (env) AND `feature_flags.execution_enabled: true`
(`config.yaml`) are set. With either switch in its default position the
engine is in dry-run — order intents are recorded and logged, never sent
(`execution/propr_client.py`, wrapping the vendored SDK in
`execution/vendor/propr_sdk.py`).

> **Note:** there is no `execution/propr_stub.py`. That original Stage 1
> stub (which raised `NotImplementedError`) has been replaced by the
> dry-run-gated `execution/propr_client.py`; the safety guarantee is now
> the two-switch gate, not the absence of an order path. Older docs that
> still reference `propr_stub.py` predate this change.

Deployed live to Railway as a worker — see "Live deployment" below. The
pytest suite is the acceptance contract (see [Tests](#tests)).

Built from [`btc-signal-bot-build-spec.md`](btc-signal-bot-build-spec.md).
See [`docs/RESEARCH_FINDINGS.md`](docs/RESEARCH_FINDINGS.md) for every
cited source behind the API/indicator choices (Fisher period 10 per
Ehlers' primary paper, OBV + 20-SMA confirmation, Hyperliquid
candleSnapshot feed), and
[`docs/STRATEGY_PSEUDOCODE.md`](docs/STRATEGY_PSEUDOCODE.md) for the
decision tree the code implements.

**This strategy is live and unvalidated — no proven edge exists.** A
walk-forward backtest harness (`backtest.py`) and a full research program
(`docs/RESEARCH_PROGRAM_SUMMARY.md`) DO exist, but their results are
SIMULATED (idealized touch fills, no slippage/funding) and found no
statistically provable edge in the reachable candle data. Parameters were
chosen conservatively (cited defaults where they exist) but there is no
validated live performance evidence behind them. Treat every alert as an
untested hypothesis, not a proven edge.

This repo is **completely separate** from the existing Bullphoric
(ALON/TROLL/ANSEM) Telegram bot: separate Git repo, separate Telegram bot
token/channel (`BTC_SIGNAL_BOT_TELEGRAM_*` env vars), no shared code,
config, or Supabase tables.

## Strategy summary

- **4H bias** (`strategy/bias_4h.py`): fractal swing detection →
  Fibonacci retracement/extension + horizontal S/R → BULLISH / BEARISH /
  NEUTRAL.
- **1H trigger** (`strategy/trigger_1h.py`): Fisher Transform (period 10,
  Ehlers' primary-source default — supersedes the platform-convention 9)
  cross + OBV-vs-its-own-20-SMA confirmation, evaluated only on closed 1H
  candles.
- **Confluence gate** (`strategy/signals.py`): a signal only exists when
  the 1H trigger direction matches the 4H bias. Stop is structural
  (beyond the relevant S/R/swing), target is the next opposing structural
  level, and any signal with R:R < 2:1 is suppressed and logged, not
  alerted.
- **Sizing** (`risk/sizing.py`): 0.75% equity risk per trade by default,
  truncated down to Hyperliquid's BTC quantity step.
- **Circuit breaker** (`risk/circuit_breaker.py`): halts new signals at
  −2.5% daily P&L (a deliberate buffer inside Propr's real 3% daily
  limit), resumes automatically at 00:00 UTC.
- **Ledger** (`ledger/tracker.py`): tracks the bot's own hypothetical
  open/closed positions so the Telegram channel is a complete auditable
  record, and so the circuit breaker has a real equity curve to gate on.

## Repository layout

```
config.yaml            # account params, risk %, telegram, feature flags
strategy/               # bias_4h.py, trigger_1h.py, signals.py — zero imports from alerts/ or execution/
data/feed.py            # Hyperliquid public OHLCV fetch
risk/                   # sizing.py, circuit_breaker.py, gate.py, challenge.py
alerts/                 # telegram.py, formats.py
execution/              # propr_client.py (dry-run-gated Propr client) + vendor/propr_sdk.py
db/                     # store.py (Postgres telemetry + live settings), schema.sql
ledger/tracker.py       # hypothetical positions + daily P&L
telegram_control/       # Telegram control-plane process (/settings, /risk, /run, trade panel) — `python -m telegram_control`
guardian.py             # independent Prop-Saver watchdog process (soft-halt / hard-flatten) — `python guardian.py`
main.py                 # Stage 2 scheduler loop (candle-close driven)
tests/                  # pytest unit tests against synthetic data
```

`strategy/`, `risk/`, and `ledger/` have zero imports from `alerts/` or
`execution/` — signals are plain data objects. This is what lets Stage 2
bolt an execution layer on without touching strategy code.

## Live deployment (Railway)

Deployed 2026-07-07 as a Railway **worker** (see `Procfile`) in its own
project — completely separate from Bullphoric's Railway service:

- Project/service: `btc-signal-bot` (workspace: zanos88's Projects)
- Env vars (service-scoped, isolated): `BTC_SIGNAL_BOT_TELEGRAM_TOKEN`,
  `BTC_SIGNAL_BOT_TELEGRAM_CHAT_ID` (note: channel IDs need the `-100`
  prefix — `-1004401790873`, not the bare ID BotFather-adjacent tools show)
- Bot: @Tradingdeskincbot → channel "TradingDesk Inc (Propr)"
- On boot the bot sends a startup heartbeat immediately, so a successful
  deploy is visible in the channel within seconds

Operate it with the Railway CLI from this directory:

```bash
railway logs --service btc-signal-bot    # read live logs ("alive:" lines = healthy)
railway up --service btc-signal-bot --detach   # redeploy after changes
# stop/restart/scale: Railway dashboard -> project btc-signal-bot
```

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in BTC_SIGNAL_BOT_TELEGRAM_TOKEN / _CHAT_ID
export $(cat .env | xargs)   # or use your process manager's env loading
python main.py
```

The bot polls every 60 seconds but only evaluates strategy logic when a
new trigger-timeframe candle has closed — detected data-drivenly via
`last_trigger_open_seen` / `newest_closed_open_time` in `main.py` (robust
for any native interval, no epoch-boundary assumptions). Logs go to
stdout (`logging.basicConfig`); redirect to a file or your platform's log
capture as needed.

### Stopping

`Ctrl+C` (SIGINT) or send SIGTERM to the process — the **trading state**
(the `Ledger`'s open/closed positions and equity curve, and the
`CircuitBreaker`) is in-memory only, so a restart begins a fresh
day/equity curve from `config.yaml`'s `starting_equity_usd`. This is a
known Stage 1 limitation (see Open Items). Note that the persisted equity
high-water mark (`equity_hwm` table) and the DB-backed engine settings
(`engine_state`, `risk_params`, `indicator_config`, `strategy_settings`)
DO survive a restart — only the paper P&L baseline resets.

### Configuring

Edit `config.yaml` for the values the running engine actually consumes:
account size (`account.starting_equity_usd`), the BTC quantity step
(`risk.btc_sz_decimals`), the 4H-bias `strategy.fractal_width` /
`strategy.sr_lookback`, the polled `data.coin`, the heartbeat interval,
and the `feature_flags` execution switch. **Note:** the running engine
reads active timeframes, indicator toggles, live risk params, and signal
geometry from Postgres (`strategy_settings`, `indicator_config`,
`risk_params` — set via the Telegram `/settings` menu), *not* from
`config.yaml`; in the `data:` block only the commented `timeframes` /
`strategy_mode` lines are DB-backed and unread — `data.coin` IS read by
the engine (it is the polled symbol, `main.py`). Likewise `risk.risk_pct`,
`strategy.fisher_period`, `strategy.obv_sma_period`, and
`strategy.min_reward_risk` are **documentation of the code's defaults, not
live inputs** — the effective `risk_pct` comes from the DB `risk_params`,
and the indicator periods and R:R minimum are module constants
(`strategy/trigger_1h.py`, `strategy/signals.py`); editing those keys
alone changes nothing. **Never lower `risk.circuit_breaker_halt_pct` below
−2.5%** (build spec section 6/11 — it's a hard-coded constant in
`risk/circuit_breaker.py`, not read from config, specifically so it can't
be casually widened).

### Reading logs / alerts

- Console/stdout: `INFO` level per-loop status, `WARNING` on feed or
  Telegram-send failures (never a silent `except: pass`).
- Telegram channel: six alert types — entry signal, exit (stop/target),
  daily summary (00:00 UTC), heartbeat (every 4h — silence means the
  process is dead), regime shift (T2, on any 4H bias change, silent), and
  circuit-breaker halt (on the −2.5% daily-P&L trip).

## Tests

```bash
python -m pytest -q
```

The suite is **262 tests across 32 files** (251 passing, 11 skipped in a
default environment) covering the strategy, risk, ledger, alerts,
execution-client and data layers against synthetic data (no live API
calls). The 11 skips are the DB-backed suites that need
`TEST_DATABASE_URL` (`tests/test_db_trigger.py`,
`tests/test_forward_report.py` — see below) plus the web-dashboard tests
(`tests/test_web_dashboard.py`, `tests/test_confluence_insight.py`) that
need the optional `web/requirements.txt` deps. These are the
acceptance contract: if a change breaks one, fix the module, not the
test.

### DB trigger tests (staging Supabase only)

`tests/test_db_trigger.py` needs a real Postgres and is **skipped** by the
command above. It runs ONLY against the staging project
`btc-signal-bot-staging`, via `TEST_DATABASE_URL` (the staging session-pooler
URI) — it never reads `DATABASE_URL` and never touches the live engine's DB:

```powershell
$env:TEST_DATABASE_URL = "postgresql://postgres.<ref>:<pw>@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
python -m pytest tests/test_db_trigger.py -v   # 7 passed
```

See `docs/V2_RUNBOOK.md` → "Running the DB tests" for the rationale (the
2026-07-08 live-engine-paused incident) and the live-ref guard.

## Stage 2 integration notes (execution layer — partially built)

The Propr execution layer now exists as `execution/propr_client.py`
(dry-run-gated) wrapping the vendored SDK, and `main.py` already wires the
entry-signal branch to `create_entry_with_bracket(...)` on an ACTIVE
engine state — all behind the two-switch dry-run gate. Strategy code was
not touched (module firewall preserved). Remaining work to go fully live:

1. **Resolve the account-equity endpoint gap first** — no confirmed Propr
   endpoint returns live total equity (see Open Items below). `main.py`
   currently derives "equity" purely from the hypothetical ledger; going
   live must decide whether to trust Propr's real equity or keep
   paper-tracking in parallel for reconciliation.
2. Feed live Propr position/fill data into `ledger/tracker.py` in place of
   the price-touch simulation, so exits reflect real fills, not idealized
   stop/target execution.
3. Add reconciliation logging comparing Propr's reported equity against
   the ledger's tracked equity — any divergence should be a WARNING per
   this project's no-silent-fallback rule.
4. Harden the order-dispatch error paths before flipping either switch —
   see the code-health notes (`create_entry_with_bracket`/`kill_sequence`
   have no per-item retry/rollback today).

## Open Items / Needs Manual Confirmation

Full detail in `docs/RESEARCH_FINDINGS.md`. Consolidated:

1. **Propr account/equity endpoint** — not found in public docs. Blocks
   Stage 2's equity-based sizing/circuit-breaker wiring against the real
   account; does not block Stage 1.
2. **Propr WebSocket schema** — mentioned but undocumented. Only matters
   for Stage 2 push-based position updates.
3. **OBV moving-average window (currently 20)** — this repo's own
   reasonable default, not sourced from a citation. Revisit after any
   forward-testing period.
4. **Hyperliquid `/info` public rate limit** — not explicitly published.
   Current candle-close-driven polling cadence is conservative but
   unverified against a hard number.
5. **No persisted state across restarts** — Stage 1's `Ledger` and
   `CircuitBreaker` are in-memory only. A process restart mid-day resets
   the daily P&L baseline. Acceptable for a Stage 1 pilot; Stage 2 (or an
   earlier Stage 1 hardening pass) should persist ledger state (e.g. to a
   local SQLite file) if extended unattended uptime is required.

## Risk & scope reminders

- Execution is dry-run by default (two-switch gate — see the top of this
  README). No performance claims are made or implied anywhere in this repo.
- Prohibited by the Propr challenge terms and not present here: HFT,
  martingale, or grid logic.
- All strategy decisions evaluate on closed candles only (no
  intra-candle/repainting signals).
