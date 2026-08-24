"""Trend dry-run forward test — paper-only, BTC 1D long/flat.

Forward-tests the round-3/4 tournament survivors (docs/STRATEGY_TOURNAMENT.md
section 10, protocol in docs/TREND_FORWARD_TEST.md) with zero capital risk:

    tsmom30  (primary)  long while close > close 30 daily bars ago
    sma50    (shadow)   long while close > SMA(50)
    buy_hold (benchmark) always long

Paper equity $100,000 per track, taker fee 0.075% per side on every position
change, marking convention identical to the tournaments: the position decided
at the close of bar j earns bar j+1's log return; the fee lands on the flip
bar. Marks go to the `trend_forward_marks` table ONLY — this process never
writes portfolio_telemetry / trade_execution_ledger / engine_state /
strategy_settings / risk_params (the floor-guard trigger reads
portfolio_telemetry's latest row unfiltered; see db/schema.sql).

Tick-and-exit and idempotent: each run fetches the last 300 closed 1D bars,
recomputes the deterministic position series, and inserts one mark per track
per unprocessed bar (UNIQUE + ON CONFLICT DO NOTHING). Reruns and overlapping
schedulers are no-ops; downtime up to ~270 days self-heals on the next run.

Usage:
    railway run --service btc-signal-bot python forward_test.py --once
    python forward_test.py --report            # read-only summary
    python forward_test.py --loop              # resident mode (future service)

Telegram: flips post audibly to the live channel tagged [TREND-FWD paper];
inception is silent. --no-telegram (or missing env) disables sends.
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime, timezone

from bisect import bisect_right

from alerts.telegram import TelegramClient, TelegramConfigError
from data.feed import fetch_candles
from db.store import TelemetryStore
from strategy.timeframes import interval_seconds
from strategy.trend_rules import sma_positions, tsmom_positions
from strategy.trigger_1h import fisher_transform, sma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forward_test")

SYMBOL = "BTC"
TF = "1d"
LOOKBACK_BARS = 300
MIN_HISTORY = 60          # bars of history required before a bar may be marked
START_EQUITY = 100_000.0
FEE = 0.00075             # taker 0.075% per side, on |position change|
POLL_SECONDS = 300
TAG = "[TREND-FWD paper]"

STRATEGIES = {
    "tsmom30": lambda cs: tsmom_positions(cs, 30),
    "sma50": lambda cs: sma_positions(cs, 50),
    "buy_hold": lambda cs: [1] * len(cs),
}

# ── Track 4 mean-reversion track (4H trigger / 12H bias) ────────────────
# Design frozen by Round 4 / Track 4-Comp / Study 2 / Round 6
# (docs/TRACK4_UNCONSTRAINED_MEAN_REVERSION.md): long-only, 4H Fisher
# <= -1.25 during a 12H-SMA30 uptrend, exit on first profitable close,
# no stop, no cap. This reimplements that same validated rule as a 0/1
# position series (matching tsmom30/sma50's shape) so it can share the
# tick loop's bar-by-bar marking/fee/flip-alert machinery unchanged;
# scripts/track4_mean_reversion.py remains the source of truth for the
# backtest sweeps this was validated against.
TRACK4_NAME = "track4_meanrev"
TRACK4_TF = "4h"
TRACK4_BIAS_TF = "12h"
TRACK4_THRESHOLD = 1.25
TRACK4_SMA_WINDOW = 30
TRACK4_REVERSAL_EXIT_LEVEL = 1.5
TRACK4_LOOKBACK_BARS = 300      # ~50 days of 4H; max observed hold was 16.2 days
TRACK4_BIAS_LOOKBACK_BARS = 300  # ~150 days of 12H, ample warmup for SMA30
TRACK4_MIN_HISTORY = 60         # matches WARMUP_4H in track4_mean_reversion.py


def track4_bias_dirs(bias_candles) -> tuple[list[int], list[int]]:
    closes = [c.close for c in bias_candles]
    s = sma(closes, TRACK4_SMA_WINDOW)
    dirs = [0 if i < TRACK4_SMA_WINDOW - 1
            else (1 if closes[i] > s[i] else (-1 if closes[i] < s[i] else 0))
            for i in range(len(closes))]
    return dirs, [c.close_time_ms for c in bias_candles]


def track4_meanrev_positions(candles_4h, bias_dirs, bias_close_ms) -> list[int]:
    """0/1 position series: 1 while a long mean-reversion trade is open.
    Long-only, entry Fisher<=-1.25 AND 12H-SMA30 uptrend, exit on first
    profitable close (net of round-trip fee) or Fisher back through +1.5,
    no stop, no cap — see the module docstring above."""
    fisher = fisher_transform(candles_4h)[0]
    pos = [0] * len(candles_4h)
    in_trade = False
    entry_price = 0.0
    for i in range(TRACK4_MIN_HISTORY, len(candles_4h)):
        c = candles_4h[i]
        if in_trade:
            net = (c.close / entry_price - 1) - 2 * FEE
            if net > 0 or fisher[i] >= TRACK4_REVERSAL_EXIT_LEVEL:
                in_trade = False
                pos[i] = 0
            else:
                pos[i] = 1
            continue
        bj = bisect_right(bias_close_ms, c.close_time_ms) - 1
        b = bias_dirs[bj] if bj >= 0 else 0
        if fisher[i] <= -TRACK4_THRESHOLD and b == 1:
            in_trade = True
            entry_price = c.close
            pos[i] = 1
    return pos


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def last_mark(conn, strategy: str):
    return conn.execute(
        "SELECT bar_open_time_ms, equity FROM trend_forward_marks "
        "WHERE strategy = %s AND symbol = %s ORDER BY bar_open_time_ms DESC LIMIT 1",
        (strategy, SYMBOL),
    ).fetchone()


def insert_mark(conn, strategy: str, candle, position: int, net: float,
                equity: float, flipped: bool) -> bool:
    """Returns True only when the row was actually inserted (not a rerun)."""
    cur = conn.execute(
        "INSERT INTO trend_forward_marks "
        "(bar_open_time_ms, bar_close_utc, strategy, symbol, close, position,"
        " bar_log_return, equity, flipped) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (strategy, bar_open_time_ms) DO NOTHING",
        (candle.open_time_ms, _ms_to_utc(candle.close_time_ms), strategy, SYMBOL,
         candle.close, position, net, equity, flipped),
    )
    return cur.rowcount == 1


def tick(store: TelemetryStore, telegram: TelegramClient | None) -> bool:
    """Process any unmarked closed bars. Returns True when >=1 mark was
    written (used by --loop to rate-limit the silent report)."""
    now_ms = int(time.time() * 1000)
    span = LOOKBACK_BARS * interval_seconds(TF) * 1000
    candles = fetch_candles(SYMBOL, TF, now_ms - span, now_ms)
    if len(candles) < MIN_HISTORY + 1:
        raise RuntimeError(f"only {len(candles)} candles fetched")
    conn = store._connect()
    inception_lines: list[str] = []
    wrote = False

    for name, rule in STRATEGIES.items():
        pos = rule(candles)
        lm = last_mark(conn, name)

        if lm is None:
            # Inception: start holding pos[last] from here; entry fee if long.
            j = len(candles) - 1
            p = pos[j]
            net = -FEE * p
            equity = START_EQUITY * math.exp(net)
            if insert_mark(conn, name, candles[j], p, net, equity, flipped=p != 0):
                wrote = True
                state = "LONG" if p else "FLAT"
                inception_lines.append(f"{name}: {state} @ ${candles[j].close:,.0f}")
                logger.info("inception %s %s equity=%.2f", name, state, equity)
            continue

        last_open, equity = lm[0], float(lm[1])
        if last_open < candles[MIN_HISTORY].open_time_ms:
            logger.error("%s: gap exceeds candle retention (last mark %s) — "
                         "manual restart of the forward test required", name, last_open)
            continue
        for j in range(1, len(candles)):
            if candles[j].open_time_ms <= last_open:
                continue
            r = math.log(candles[j].close / candles[j - 1].close)
            delta = abs(pos[j] - pos[j - 1])
            net = pos[j - 1] * r - FEE * delta
            equity *= math.exp(net)
            flipped = delta != 0
            if insert_mark(conn, name, candles[j], pos[j - 1], net, equity, flipped):
                wrote = True
                if flipped and telegram is not None and name != "buy_hold":
                    pnl = (equity / START_EQUITY - 1) * 100
                    state = "LONG" if pos[j] else "FLAT"
                    telegram.send(
                        f"<b>{TAG} {name} FLIP → {state}</b> @ ${candles[j].close:,.0f}"
                        f" | paper equity ${equity:,.0f} ({pnl:+.1f}% since inception)",
                        silent=False,
                    )
                logger.info("mark %s bar=%s pos_into=%d net=%.5f equity=%.2f%s",
                            name, _ms_to_utc(candles[j].close_time_ms).date(),
                            pos[j - 1], net, equity, " FLIP" if flipped else "")

    # Track 4 mean-reversion (4H trigger / 12H bias) — its own block since
    # it needs a second candle series, unlike the single-series STRATEGIES.
    span4h = TRACK4_LOOKBACK_BARS * interval_seconds(TRACK4_TF) * 1000
    candles_4h = fetch_candles(SYMBOL, TRACK4_TF, now_ms - span4h, now_ms)
    span12h = TRACK4_BIAS_LOOKBACK_BARS * interval_seconds(TRACK4_BIAS_TF) * 1000
    bias_candles = fetch_candles(SYMBOL, TRACK4_BIAS_TF, now_ms - span12h, now_ms)
    if len(candles_4h) >= TRACK4_MIN_HISTORY + 1 and len(bias_candles) >= TRACK4_SMA_WINDOW:
        bias_dirs, bias_close_ms = track4_bias_dirs(bias_candles)
        pos4 = track4_meanrev_positions(candles_4h, bias_dirs, bias_close_ms)
        lm4 = last_mark(conn, TRACK4_NAME)
        if lm4 is None:
            j = len(candles_4h) - 1
            p = pos4[j]
            net = -FEE * p
            equity = START_EQUITY * math.exp(net)
            if insert_mark(conn, TRACK4_NAME, candles_4h[j], p, net, equity, flipped=p != 0):
                wrote = True
                state = "LONG" if p else "FLAT"
                inception_lines.append(f"{TRACK4_NAME}: {state} @ ${candles_4h[j].close:,.0f}")
                logger.info("inception %s %s equity=%.2f", TRACK4_NAME, state, equity)
        else:
            last_open, equity = lm4[0], float(lm4[1])
            if last_open < candles_4h[TRACK4_MIN_HISTORY].open_time_ms:
                logger.error("%s: gap exceeds candle retention (last mark %s) — "
                             "manual restart of the forward test required",
                             TRACK4_NAME, last_open)
            else:
                for j in range(1, len(candles_4h)):
                    if candles_4h[j].open_time_ms <= last_open:
                        continue
                    r = math.log(candles_4h[j].close / candles_4h[j - 1].close)
                    delta = abs(pos4[j] - pos4[j - 1])
                    net = pos4[j - 1] * r - FEE * delta
                    equity *= math.exp(net)
                    flipped = delta != 0
                    if insert_mark(conn, TRACK4_NAME, candles_4h[j], pos4[j - 1], net,
                                   equity, flipped):
                        wrote = True
                        if flipped and telegram is not None:
                            pnl = (equity / START_EQUITY - 1) * 100
                            state = "LONG" if pos4[j] else "FLAT"
                            telegram.send(
                                f"<b>{TAG} {TRACK4_NAME} FLIP → {state}</b> "
                                f"@ ${candles_4h[j].close:,.0f} | paper equity "
                                f"${equity:,.0f} ({pnl:+.1f}% since inception)",
                                silent=False,
                            )
                        logger.info("mark %s bar=%s pos_into=%d net=%.5f equity=%.2f%s",
                                    TRACK4_NAME, _ms_to_utc(candles_4h[j].close_time_ms),
                                    pos4[j - 1], net, equity, " FLIP" if flipped else "")

    if inception_lines and telegram is not None:
        telegram.send(
            f"<b>{TAG} forward test inception</b> — $100k paper each, "
            f"fee 0.075%/side.\n" + "\n".join(inception_lines) +
            "\nProtocol: docs/TREND_FORWARD_TEST.md (review gates per track).",
            silent=True,
        )
    return wrote


def report_text(conn, symbol: str = SYMBOL) -> str | None:
    """The --report table as plain text (also posted silently per tick).
    Pure function of DB state — two calls on identical data must return
    identical output (regression-tested after the 2026-07-10 report-position
    confusion, which was a code change between calls, not nondeterminism)."""
    rows = conn.execute(
        "SELECT strategy, count(*), min(bar_close_utc), max(bar_close_utc), "
        "sum(CASE WHEN flipped THEN 1 ELSE 0 END) "
        "FROM trend_forward_marks WHERE symbol = %s GROUP BY strategy ORDER BY strategy",
        (symbol,),
    ).fetchall()
    if not rows:
        return None
    lines = [f"{'strategy':10s} {'marks':>5s} {'flips':>5s} {'equity':>12s} {'net%':>8s} "
             f"{'pos':>4s} {'MAE%':>7s}  window"]
    for strategy, n, first, last_ts, flips in rows:
        eq_row = conn.execute(
            "SELECT equity, position, flipped FROM trend_forward_marks "
            "WHERE strategy = %s AND symbol = %s ORDER BY bar_open_time_ms DESC LIMIT 1",
            (strategy, symbol),
        ).fetchone()
        equity, position, flipped = float(eq_row[0]), eq_row[1], eq_row[2]
        # Display the CURRENT position (held from the last close onward).
        # Normal marks store the position held INTO the bar, so a flip at
        # that close means current = 1 - position; the inception row (n == 1)
        # already stores the position held FROM inception.
        current = position if n == 1 else ((1 - position) if flipped else position)
        net = (equity / START_EQUITY - 1) * 100
        mae_s = "-"
        if current and strategy == TRACK4_NAME:
            # This strategy's real risk lives in underwater depth, not win
            # rate (docs/TRACK4_UNCONSTRAINED_MEAN_REVERSION.md) — surface
            # current MAE on every report while a position is open, not
            # just on entry/exit. Entry = the most recent flip-into-LONG
            # mark (position=1, flipped=true, or the inception row itself
            # if it started LONG); MAE = worst close since then vs entry.
            entry_row = conn.execute(
                # Entry anchor = the most recent flip (into LONG, since the
                # position is currently open) OR the inception mark. NOTE: a
                # RUNNING entry marks position=0/flipped=true (position held
                # INTO the entry bar was flat), so this must NOT filter
                # position=1 — that would only catch inception-LONG entries.
                "SELECT bar_open_time_ms, close FROM trend_forward_marks "
                "WHERE strategy = %s AND symbol = %s AND "
                "(flipped = true OR bar_open_time_ms = "
                " (SELECT min(bar_open_time_ms) FROM trend_forward_marks "
                "  WHERE strategy = %s AND symbol = %s)) "
                "ORDER BY bar_open_time_ms DESC LIMIT 1",
                (strategy, symbol, strategy, symbol),
            ).fetchone()
            if entry_row:
                entry_open_ms, entry_close = entry_row[0], float(entry_row[1])
                min_close = conn.execute(
                    "SELECT min(close) FROM trend_forward_marks WHERE strategy = %s "
                    "AND symbol = %s AND bar_open_time_ms >= %s",
                    (strategy, symbol, entry_open_ms),
                ).fetchone()[0]
                if min_close is not None:
                    mae_s = f"{(float(min_close) / entry_close - 1) * 100:+.2f}"
        lines.append(f"{strategy:10s} {n:5d} {flips:5d} {equity:12,.2f} {net:+7.2f}% "
                     f"{('LONG' if current else 'FLAT'):>4s} {mae_s:>7s}  "
                     f"{first:%Y-%m-%d} .. {last_ts:%Y-%m-%d}")
    return "\n".join(lines)


def report(store: TelemetryStore) -> None:
    print(report_text(store._connect()) or "no marks yet — run --once first")


def send_report(store: TelemetryStore, telegram: TelegramClient | None) -> None:
    """Silent per-tick status post — flips remain the only audible messages."""
    if telegram is None:
        return
    text = report_text(store._connect())
    if text:
        telegram.send(f"<b>{TAG} report</b>\n<pre>{text}</pre>", silent=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="single tick and exit")
    mode.add_argument("--loop", action="store_true", help="resident mode (poll %ds)" % POLL_SECONDS)
    mode.add_argument("--report", action="store_true", help="read-only summary")
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args()

    store = TelemetryStore()
    if args.report:
        report(store)
        return
    store.apply_schema()

    telegram: TelegramClient | None = None
    if not args.no_telegram:
        try:
            telegram = TelegramClient()
        except TelegramConfigError:
            logger.warning("Telegram env not set — running silent")

    if args.once:
        tick(store, telegram)
        report(store)
        send_report(store, telegram)  # every scheduled tick (<=3/day) posts silently
        return
    logger.info("resident loop, poll %ds", POLL_SECONDS)
    while True:
        try:
            # Loop mode polls every 5 min; report only on ticks that wrote
            # marks so a resident deployment can't emit 288 posts/day.
            if tick(store, telegram):
                send_report(store, telegram)
        except Exception:
            logger.warning("tick failed; retrying next poll", exc_info=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
