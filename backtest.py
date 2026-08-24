"""Walk-forward backtest — replays Hyperliquid history through the EXACT
live strategy code (strategy.signals.evaluate_signal, same edge-trigger
alignment as main.py, same R:R gate, same 300-bar slices the engine uses).

HONESTY CONTRACT (print + store with every run):
- SIMULATED results. Fills are idealized at the exact touch price; real
  fills have slippage and funding costs that are NOT modeled.
- Candle-granularity ambiguity: if one candle touches BOTH stop and
  target, the STOP is assumed to fill first (conservative).
- Taker fees modeled at 0.075% per side (verified Propr rate).
- History window is capped by Hyperliquid's 5,000-candle retention on
  the trigger timeframe (~208 days for 1h; ~2.3yr for 4h).
- Past confluence behavior does not promise future results; this data
  informs indicator review, it does not "validate" the strategy.

Usage:
    # single run
    railway run --service btc-signal-bot python backtest.py \
        [--bias-tf 4h] [--trigger-tf 1h] [--indicators default|all|csv] \
        [--stop-model structural|hybrid] [--atr-multiplier 1.5] [--no-store]

    # config-driven batch sweep (one stored run per combo)
    railway run --service btc-signal-bot python backtest.py --sweep sweep_config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import math
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone

import yaml
from ulid import ULID

from data.feed import Candle, fetch_candles, fetch_funding_history
from strategy.signals import (
    DEFAULT_ATR_MULTIPLIER,
    DEFAULT_BLUE_SKY_ATR_MULTIPLIER,
    DEFAULT_INDICATOR_CONFIG,
    FISHER4H_EXHAUSTION_THRESHOLD,
    INDICATOR_NAMES,
    TARGET_MODELS,
    Signal,
    SignalDirection,
    SuppressedSignal,
    evaluate_signal,
)
from strategy.counter_trend import (
    OBV_RULES,
    CounterTrendSignal,
    evaluate_counter_trend,
    opposite_cloud_edge,
)
from strategy.fisher_cycle import (
    daily_bias_at,
    is_exhausted,
    leg_stop,
    macro_broken,
    opening_direction,
)
from strategy.bias_4h import Bias, BiasResult, compute_bias
from strategy.signals import _nearest_resistance, _nearest_support
from strategy.atr import wilder_atr
from strategy.timeframes import LOOKBACK_BARS, interval_seconds, validate_combo
from strategy.trigger_1h import fisher_transform

TAKER_FEE = 0.00075  # per side, verified in RESEARCH_FINDINGS Rev 3
# No-stop (spot-accumulation) favorable-exit models. The 4H bias-flip brake is
# NOT one of these — it is the standing catastrophe floor under every model.
NO_STOP_EXIT_MODELS = ("first_profit", "fib_target", "resistance_rejection",
                       "min_move_first_profit", "trailing_once_profitable")
WARMUP_TRIGGER_BARS = 40      # fisher(10) + obv sma(20) + atr(14) + margin
STANDDOWN_WINDOW_MS = 30 * 24 * 3600 * 1000  # trailing window for funding pctile / OI z
FUNDING_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__),
                                     "research", "data", "BTC_funding_history.json")


def load_funding_series(now_ms: int) -> list[tuple[int, float]]:
    """Frozen-input discipline: read the committed funding snapshot when it
    exists; otherwise fetch the full history once and freeze it."""
    if os.path.exists(FUNDING_SNAPSHOT_PATH):
        with open(FUNDING_SNAPSHOT_PATH, encoding="utf-8") as f:
            doc = json.load(f)
        return [(int(t), float(v)) for t, v in doc["rows"]]
    rows = fetch_funding_history("BTC", 0, now_ms)
    os.makedirs(os.path.dirname(FUNDING_SNAPSHOT_PATH), exist_ok=True)
    from datetime import datetime, timezone
    doc = {"coin": "BTC", "source": "hyperliquid fundingHistory (hourly)",
           "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "row_count": len(rows), "rows": rows}
    with open(FUNDING_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return rows


OI_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__),
                                "research", "data", "BTC_oi_history.json")


def load_oi_series() -> list[tuple[int, float]] | None:
    """Frozen OI history (Coinalyze, hourly) for the stand-down conjunction,
    or None when no snapshot exists. Coverage is limited by the source's
    free-tier retention — see docs/OI_LIQUIDATION_PHASE0_PHASE1.md; bars
    outside coverage get oi_z=None and the conjunction declines to fire."""
    if not os.path.exists(OI_SNAPSHOT_PATH):
        return None
    with open(OI_SNAPSHOT_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    return [(int(t), float(v)) for t, v in doc["rows"]]
WARMUP_BIAS_BARS = 120        # fractal/S-R structure + ichimoku(52+disp)


@dataclass
class TradeResult:
    entry_ts: datetime
    exit_ts: datetime | None
    direction: str
    entry: float
    stop: float
    target: float
    reward_risk: float
    exit_reason: str  # target | stop | unresolved | reversion | bias_flip
    gross_r: float | None
    net_r: float | None
    bars_held: int
    indicators_snapshot: dict
    # patient-hold variant only (no-stop accumulation): deepest adverse
    # excursion while held, as a fraction of entry and in R (vs the never-
    # placed structural stop). None for stop/target trades.
    mae_frac: float | None = None
    mae_r: float | None = None


def _simulate_no_stop_exit(candles: list[Candle], entry_index: int, signal: Signal,
                           is_long: bool, risk: float, entry_ts: datetime,
                           bias4h_series: list[tuple[int, Bias]] | None,
                           exit_model: str = "first_profit",
                           sr4h_series: list[tuple[int, BiasResult]] | None = None,
                           min_move_r: float = 1.0, trail_arm_r: float = 1.0,
                           trail_dist_r: float = 1.0) -> TradeResult:
    """No-stop exit family (BACKTEST-ONLY, spot-capital accumulation — NOT
    comp/Propr). The structural stop is NEVER placed: `risk = |entry - stop|`
    only expresses P&L and MAE in R against that never-placed stop. The 4H
    bias-flip is the standing catastrophe brake under EVERY model (`bias_flip`;
    NEUTRAL counts as invalidation, per Track 3's `macro_broken`).

    Favorable exit per `exit_model`:
      first_profit             first net-profitable CLOSE (`reversion`)
      fib_target               touch of the entry's fib-extension target
      resistance_rejection     touch nearest structural R/S that fails to close
                               beyond it (real rejection); closing through runs on
      min_move_first_profit    first_profit, but only after >= min_move_r R favorable
      trailing_once_profitable trail trail_dist_r R behind the best price, armed
                               once >= trail_arm_r R favorable (pre-arm: brake only)

    Deepest adverse excursion (MAE) is tracked intrabar (long: lows, short:
    highs) and reported in % and R with P&L-equal prominence. A favorable exit
    takes precedence over the brake within a bar. Runs out of data ->
    `unresolved` (excluded from win/loss)."""
    if bias4h_series is None:
        raise ValueError("no-stop exit requires bias4h_series")
    if exit_model not in NO_STOP_EXIT_MODELS:
        raise ValueError(f"unknown exit_model {exit_model!r} — allowed: {NO_STOP_EXIT_MODELS}")
    if exit_model == "resistance_rejection" and sr4h_series is None:
        raise ValueError("resistance_rejection requires sr4h_series")
    opened_bias = Bias.BULLISH if is_long else Bias.BEARISH
    bias_times = [t for t, _ in bias4h_series]
    sr_times = [t for t, _ in sr4h_series] if sr4h_series else []
    sign = 1 if is_long else -1
    r_frac = risk / signal.entry if signal.entry else 0.0   # 1R as a fraction of entry
    worst_frac = 0.0            # most-negative adverse excursion (fraction of entry)
    best_fav = 0.0             # max favorable excursion (fraction of entry)
    best_price = signal.entry  # extreme in favor, for trailing
    armed = False

    def _mae_r() -> float | None:
        return (worst_frac * signal.entry / risk) if risk else None

    def _result(j: int, exit_price: float, reason: str) -> TradeResult:
        gross_r = sign * (exit_price - signal.entry) / risk if risk else 0.0
        fee_r = (signal.entry + exit_price) * TAKER_FEE / risk if risk else 0.0
        return TradeResult(
            entry_ts=entry_ts,
            exit_ts=datetime.fromtimestamp(candles[j].close_time_ms / 1000, tz=timezone.utc),
            direction=signal.direction.value, entry=signal.entry, stop=signal.stop,
            target=signal.target, reward_risk=signal.reward_risk, exit_reason=reason,
            gross_r=gross_r, net_r=gross_r - fee_r, bars_held=j - entry_index,
            indicators_snapshot={}, mae_frac=worst_frac, mae_r=_mae_r(),
        )

    for j in range(entry_index + 1, len(candles)):
        c = candles[j]
        adverse = ((c.low - signal.entry) / signal.entry if is_long
                   else (signal.entry - c.high) / signal.entry)
        worst_frac = min(worst_frac, adverse)
        fav = ((c.high - signal.entry) / signal.entry if is_long
               else (signal.entry - c.low) / signal.entry)
        best_fav = max(best_fav, fav)

        # ── favorable exit (model-specific; precedence over the brake) ──
        if exit_model in ("first_profit", "min_move_first_profit"):
            if (exit_model == "min_move_first_profit" and not armed
                    and r_frac and best_fav >= min_move_r * r_frac):
                armed = True
            ready = exit_model == "first_profit" or armed
            if ready and sign * (c.close - signal.entry) / signal.entry - 2 * TAKER_FEE > 0:
                return _result(j, c.close,
                               "reversion" if exit_model == "first_profit" else "min_move_profit")
        elif exit_model == "fib_target":
            tgt = signal.target
            if tgt is not None and ((c.high >= tgt) if is_long else (c.low <= tgt)):
                return _result(j, tgt, "fib_target")
        elif exit_model == "resistance_rejection":
            si = bisect_right(sr_times, c.close_time_ms) - 1
            if si >= 0:
                br = sr4h_series[si][1]
                prev_close = candles[j - 1].close
                if is_long:
                    lvl = _nearest_resistance(br, prev_close)
                    if lvl is not None and c.high >= lvl and c.close < lvl:
                        return _result(j, c.close, "resistance_rejection")
                else:
                    lvl = _nearest_support(br, prev_close)
                    if lvl is not None and c.low <= lvl and c.close > lvl:
                        return _result(j, c.close, "resistance_rejection")
        elif exit_model == "trailing_once_profitable":
            if armed:
                trail = (best_price - trail_dist_r * risk if is_long
                         else best_price + trail_dist_r * risk)
                if (c.low <= trail) if is_long else (c.high >= trail):
                    return _result(j, trail, "trailing_stop")
            elif r_frac and best_fav >= trail_arm_r * r_frac:
                armed = True                       # takes effect from the next bar
        # update the favorable extreme AFTER the hit check (arming bar sets the base)
        best_price = max(best_price, c.high) if is_long else min(best_price, c.low)

        # ── catastrophe brake: 4H bias flipped off the trade's direction ──
        bi = bisect_right(bias_times, c.close_time_ms) - 1
        if bi >= 0 and bias4h_series[bi][1] != opened_bias:
            return _result(j, c.close, "bias_flip")

    return TradeResult(
        entry_ts=entry_ts, exit_ts=None, direction=signal.direction.value,
        entry=signal.entry, stop=signal.stop, target=signal.target,
        reward_risk=signal.reward_risk, exit_reason="unresolved",
        gross_r=None, net_r=None, bars_held=len(candles) - 1 - entry_index,
        indicators_snapshot={}, mae_frac=worst_frac, mae_r=_mae_r(),
    )


def simulate_outcome(
    candles: list[Candle],
    entry_index: int,
    signal: Signal,
    fisher4h_exit: bool = False,
    fisher4h_series: list[tuple[int, float]] | None = None,
    exhaustion_threshold: float = FISHER4H_EXHAUSTION_THRESHOLD,
    patient_hold_exit: bool = False,
    bias4h_series: list[tuple[int, Bias]] | None = None,
    exit_model: str = "first_profit",
    sr4h_series: list[tuple[int, BiasResult]] | None = None,
    min_move_r: float = 1.0,
    trail_arm_r: float = 1.0,
    trail_dist_r: float = 1.0,
) -> TradeResult:
    """Walk candles AFTER entry until stop or target is TOUCHED (high/low,
    not close). Both touched in one candle -> stop first (conservative).
    Runs out of data -> unresolved (excluded from win/loss stats).

    fisher4h_exit (BACKTEST-ONLY variant): additionally exit at a trigger
    bar's CLOSE when the 4H Fisher crosses INTO extended territory in the
    trade's favor (LONG: F crosses >= +threshold; SHORT: <= -threshold) —
    the move you're in looks exhausted, scale out ahead of the structural
    target. Edge semantics: already-extended-at-entry never fires until
    Fisher leaves and re-enters the extended zone. Stop/target touches in
    the same bar take precedence (conservative). fisher4h_series is
    [(close_time_ms, fisher_value), ...] for CLOSED 4h bars, ascending —
    the Fisher construction is causal, so a precomputed series is
    lookahead-safe."""
    if fisher4h_exit and not fisher4h_series:
        raise ValueError("fisher4h_exit requires fisher4h_series")
    is_long = signal.direction == SignalDirection.LONG
    risk = abs(signal.entry - signal.stop)
    entry_ts = datetime.fromtimestamp(candles[entry_index].close_time_ms / 1000, tz=timezone.utc)

    if patient_hold_exit:
        return _simulate_no_stop_exit(candles, entry_index, signal, is_long, risk,
                                      entry_ts, bias4h_series, exit_model=exit_model,
                                      sr4h_series=sr4h_series, min_move_r=min_move_r,
                                      trail_arm_r=trail_arm_r, trail_dist_r=trail_dist_r)

    def _in_favor(f: float) -> bool:
        return f >= exhaustion_threshold if is_long else f <= -exhaustion_threshold

    def _result(j: int, exit_price: float, reason: str) -> TradeResult:
        sign = 1 if is_long else -1
        gross_r = sign * (exit_price - signal.entry) / risk
        fee_r = (signal.entry + exit_price) * TAKER_FEE / risk
        return TradeResult(
            entry_ts=entry_ts,
            exit_ts=datetime.fromtimestamp(candles[j].close_time_ms / 1000, tz=timezone.utc),
            direction=signal.direction.value, entry=signal.entry, stop=signal.stop,
            target=signal.target, reward_risk=signal.reward_risk, exit_reason=reason,
            gross_r=gross_r, net_r=gross_r - fee_r, bars_held=j - entry_index,
            indicators_snapshot={},
        )

    # index of the first 4h bar NOT yet closed at entry
    k = 0
    if fisher4h_exit:
        entry_close_ms = candles[entry_index].close_time_ms
        while k < len(fisher4h_series) and fisher4h_series[k][0] <= entry_close_ms:
            k += 1

    for j in range(entry_index + 1, len(candles)):
        c = candles[j]
        hit_stop = c.low <= signal.stop if is_long else c.high >= signal.stop
        hit_target = c.high >= signal.target if is_long else c.low <= signal.target
        if hit_stop or hit_target:
            exit_price = signal.stop if hit_stop else signal.target  # stop wins ambiguity
            return _result(j, exit_price, "stop" if hit_stop else "target")

        if fisher4h_exit:
            crossed = False
            while k < len(fisher4h_series) and fisher4h_series[k][0] <= c.close_time_ms:
                now_f = fisher4h_series[k][1]
                prev_f = fisher4h_series[k - 1][1] if k >= 1 else 0.0
                if _in_favor(now_f) and not _in_favor(prev_f):
                    crossed = True
                k += 1
            if crossed:
                return _result(j, c.close, "fisher_exhaustion")

    return TradeResult(entry_ts=entry_ts, exit_ts=None, direction=signal.direction.value,
                       entry=signal.entry, stop=signal.stop, target=signal.target,
                       reward_risk=signal.reward_risk, exit_reason="unresolved",
                       gross_r=None, net_r=None, bars_held=len(candles) - 1 - entry_index,
                       indicators_snapshot={})


def bias_slice_no_lookahead(bias_candles: list[Candle], trigger_close_ms: int) -> list[Candle]:
    """Only bias candles CLOSED at/before the trigger close — no lookahead."""
    return [c for c in bias_candles if c.close_time_ms <= trigger_close_ms]


def log_return_stats(candles: list[Candle]) -> dict:
    """Real close-to-close log-return distribution of the series actually
    used — mean/stdev/excess kurtosis/N. Replaces the REJECTED external
    stats table from the volatility research doc with sourced numbers
    (source: the Hyperliquid candles this very run consumed)."""
    closes = [c.close for c in candles]
    rets = [math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
    n = len(rets)
    if n < 2:
        return {"n": n}
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    m2 = sum((r - mean) ** 2 for r in rets) / n
    m4 = sum((r - mean) ** 4 for r in rets) / n
    return {
        "n": n,
        "mean": mean,
        "stdev": math.sqrt(var),
        "excess_kurtosis": (m4 / (m2 * m2) - 3.0) if m2 > 0 else None,
    }


def run_backtest(bias_candles: list[Candle], trigger_candles: list[Candle],
                 config: dict, ichimoku_variant: str = "standard",
                 stop_model: str = "structural",
                 atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
                 target_model: str = "nearest_structure",
                 blue_sky_atr_multiplier: float = DEFAULT_BLUE_SKY_ATR_MULTIPLIER,
                 fisher4h_entry: bool = False,
                 fisher4h_exit: bool = False,
                 exhaustion_threshold: float = FISHER4H_EXHAUSTION_THRESHOLD,
                 candles_4h: list[Candle] | None = None,
                 standdown_entry: bool = False,
                 funding_series: list[tuple[int, float]] | None = None,
                 funding_pctile_threshold: float = 85.0,
                 oi_series: list[tuple[int, float]] | None = None,
                 oi_z_min: float | None = None,
                 patient_hold_exit: bool = False,
                 bias4h_candles: list[Candle] | None = None) -> dict:
    """Walk-forward over trigger closes; mirrors main.py exactly:
    300-bar slices, edge-triggered alignment, one open trade at a time
    (max_concurrent=1, the live default).

    fisher4h_entry/fisher4h_exit are the BACKTEST-ONLY V2.2 exhaustion
    variants; both need candles_4h (a dedicated 4h series regardless of
    the TF pair — the heuristic is specifically about the 4H chart)."""
    fisher4h_series: list[tuple[int, float]] = []
    close_ms_4h: list[int] = []
    if fisher4h_entry or fisher4h_exit:
        if not candles_4h:
            raise ValueError("fisher4h variants require candles_4h")
        fisher_line = fisher_transform(candles_4h)[0]  # causal -> precompute is lookahead-safe
        close_ms_4h = [c.close_time_ms for c in candles_4h]
        fisher4h_series = list(zip(close_ms_4h, fisher_line))

    def _fisher4h_at(trigger_close_ms: int) -> float:
        """Fisher of the last 4h bar CLOSED at/before this trigger close."""
        idx = bisect_right(close_ms_4h, trigger_close_ms) - 1
        return fisher4h_series[idx][1] if idx >= 0 else 0.0

    funding_times: list[int] = []
    funding_vals: list[float] = []
    if standdown_entry:
        if not funding_series:
            raise ValueError("standdown_entry requires funding_series")
        funding_times = [t for t, _ in funding_series]
        funding_vals = [v for _, v in funding_series]
        first_eval_close = trigger_candles[WARMUP_TRIGGER_BARS].close_time_ms
        if funding_times[0] > first_eval_close - STANDDOWN_WINDOW_MS:
            raise ValueError("funding_series must start >=30d before the first evaluated bar "
                             "(causal trailing percentile needs a full window)")

    def _funding_pctile_at(trigger_close_ms: int) -> float:
        """Percentile of the latest funding print within its own TRAILING
        30-day window ending at/before this trigger close — causal by
        construction (never normalized against the full series)."""
        hi = bisect_right(funding_times, trigger_close_ms)
        lo = bisect_right(funding_times, trigger_close_ms - STANDDOWN_WINDOW_MS)
        window = funding_vals[lo:hi]
        current = funding_vals[hi - 1]
        return 100.0 * sum(1 for v in window if v <= current) / len(window)

    oi_times: list[int] = []
    oi_vals: list[float] = []
    if oi_series:
        oi_times = [t for t, _ in oi_series]
        oi_vals = [v for _, v in oi_series]

    def _oi_z_at(trigger_close_ms: int) -> float | None:
        """OI z-score vs its trailing 30-day distribution (dormant until an
        OI history source is unlocked — Phase 0 §0.2)."""
        if not oi_times:
            return None
        hi = bisect_right(oi_times, trigger_close_ms)
        lo = bisect_right(oi_times, trigger_close_ms - STANDDOWN_WINDOW_MS)
        window = oi_vals[lo:hi]
        if len(window) < 2:
            return None
        mu = sum(window) / len(window)
        sd = (sum((v - mu) ** 2 for v in window) / (len(window) - 1)) ** 0.5
        return (oi_vals[hi - 1] - mu) / sd if sd > 0 else None

    # Patient-hold (no-stop accumulation) exit needs the 4H bias as a causal
    # step function keyed by 4H close. Precompute once with the SAME trailing
    # LOOKBACK_BARS slice + compute_bias that evaluate_signal uses for entries,
    # so exit-time bias matches entry-time bias exactly (no lookahead).
    bias4h_series: list[tuple[int, Bias]] = []
    if patient_hold_exit:
        src = bias4h_candles if bias4h_candles is not None else bias_candles
        for k in range(len(src)):
            sl = src[max(0, k + 1 - LOOKBACK_BARS): k + 1]
            bias4h_series.append((src[k].close_time_ms, compute_bias(sl).bias))

    trades: list[TradeResult] = []
    suppressed = 0
    suppressed_exhaustion = 0
    suppressed_standdown = 0
    prev_alignment: str | None = None
    open_until_index = -1  # enforce one-position-at-a-time like the live gate

    for i in range(WARMUP_TRIGGER_BARS, len(trigger_candles)):
        trig_slice = trigger_candles[max(0, i + 1 - LOOKBACK_BARS): i + 1]
        bias_all = bias_slice_no_lookahead(bias_candles, trigger_candles[i].close_time_ms)
        if len(bias_all) < WARMUP_BIAS_BARS:
            continue
        bias_slice = bias_all[-LOOKBACK_BARS:]

        result, readings = evaluate_signal(
            bias_slice, trig_slice, config=config,
            ichimoku_variant=ichimoku_variant, return_readings=True,
            stop_model=stop_model, atr_multiplier=atr_multiplier,
            target_model=target_model,
            blue_sky_atr_multiplier=blue_sky_atr_multiplier,
            fisher4h_entry_filter=fisher4h_entry,
            fisher4h_value=(_fisher4h_at(trigger_candles[i].close_time_ms)
                            if fisher4h_entry else None),
            exhaustion_threshold=exhaustion_threshold,
            standdown_entry_filter=standdown_entry,
            funding_pctile_value=(_funding_pctile_at(trigger_candles[i].close_time_ms)
                                  if standdown_entry else None),
            funding_pctile_threshold=funding_pctile_threshold,
            oi_z_value=(_oi_z_at(trigger_candles[i].close_time_ms)
                        if standdown_entry else None),
            oi_z_min=oi_z_min,
        )

        enabled_votes = [r["vote"] for n, r in readings.items() if r["enabled"]]
        alignment = (enabled_votes[0] if enabled_votes
                     and all(v == enabled_votes[0] for v in enabled_votes)
                     and enabled_votes[0] != "NONE" else None)
        is_new_alignment = alignment is not None and alignment != prev_alignment
        prev_alignment = alignment

        if not is_new_alignment or i <= open_until_index:
            continue
        if isinstance(result, SuppressedSignal):
            if result.kind == "fisher4h_exhaustion":
                suppressed_exhaustion += 1
            elif result.kind == "exhaustion_standdown":
                suppressed_standdown += 1
            else:
                suppressed += 1
            continue
        if isinstance(result, Signal):
            trade = simulate_outcome(trigger_candles, i, result,
                                     fisher4h_exit=fisher4h_exit,
                                     fisher4h_series=fisher4h_series,
                                     exhaustion_threshold=exhaustion_threshold,
                                     patient_hold_exit=patient_hold_exit,
                                     bias4h_series=bias4h_series)
            trade.indicators_snapshot = readings
            trades.append(trade)
            # block new entries until this trade resolves (or forever if unresolved)
            if trade.exit_ts is not None:
                open_until_index = i + trade.bars_held
            else:
                open_until_index = len(trigger_candles)

    resolved = [t for t in trades if t.exit_reason != "unresolved"]
    wins = [t for t in resolved if t.net_r is not None and t.net_r > 0]
    losses = [t for t in resolved if t.net_r is not None and t.net_r <= 0]
    net_rs = [t.net_r for t in resolved if t.net_r is not None]

    equity_r, peak, max_dd = 0.0, 0.0, 0.0
    for r in net_rs:
        equity_r += r
        peak = max(peak, equity_r)
        max_dd = max(max_dd, peak - equity_r)

    gross_win = sum(t.net_r for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.net_r for t in losses)) if losses else 0.0

    return {
        "bars_evaluated": len(trigger_candles) - WARMUP_TRIGGER_BARS,
        "trades": trades,
        "resolved": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "unresolved": len(trades) - len(resolved),
        "suppressed_rr": suppressed,
        "suppressed_exhaustion": suppressed_exhaustion,
        "suppressed_standdown": suppressed_standdown,
        "gross_r": sum(t.gross_r for t in resolved if t.gross_r is not None),
        "net_r": sum(net_rs),
        "avg_net_r": (sum(net_rs) / len(net_rs)) if net_rs else None,
        "win_rate": (len(wins) / len(resolved)) if resolved else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "max_drawdown_r": max_dd,
    }


# ── counter-trend (Track 2) — a SEPARATE strategy path, same tables ──

CT_ICHIMOKU_WINDOW = 120   # enough history for the standard cloud (52+26) + margin
CT_EXHAUSTION_WINDOW = 15  # bars to look back for the Fisher exhaustion extreme.
# The exhaustion PRECEDES the reclaim entry by a swing's worth of bars, so
# the extreme is checked over this window, NOT the 6-bar cross window
# (a diagnostic on 4h/1h showed the entry-bar / 6-bar Fisher gate fired on
# 0 of 73 valid E2E-geometry bars — the oversold low sits ~10-15 bars
# before price reclaims the cloud).


def _summarize_trades(trades: list[TradeResult], bars_evaluated: int,
                      suppressed_rr: int = 0, suppressed_exhaustion: int = 0) -> dict:
    """Same summary shape as run_backtest, reused by the counter-trend
    path so store_run / the comparison table need no special-casing."""
    resolved = [t for t in trades if t.exit_reason != "unresolved"]
    wins = [t for t in resolved if t.net_r is not None and t.net_r > 0]
    losses = [t for t in resolved if t.net_r is not None and t.net_r <= 0]
    net_rs = [t.net_r for t in resolved if t.net_r is not None]
    equity_r, peak, max_dd = 0.0, 0.0, 0.0
    for r in net_rs:
        equity_r += r
        peak = max(peak, equity_r)
        max_dd = max(max_dd, peak - equity_r)
    gross_win = sum(t.net_r for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.net_r for t in losses)) if losses else 0.0
    return {
        "bars_evaluated": bars_evaluated, "trades": trades, "resolved": len(resolved),
        "wins": len(wins), "losses": len(losses), "unresolved": len(trades) - len(resolved),
        "suppressed_rr": suppressed_rr, "suppressed_exhaustion": suppressed_exhaustion,
        "gross_r": sum(t.gross_r for t in resolved if t.gross_r is not None),
        "net_r": sum(net_rs), "avg_net_r": (sum(net_rs) / len(net_rs)) if net_rs else None,
        "win_rate": (len(wins) / len(resolved)) if resolved else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "max_drawdown_r": max_dd,
    }


def _fisher_recent_extremes(candles: list[Candle], window: int) -> tuple[float, float]:
    """(min, max) of Fisher over the last `window` closed bars of the
    given series — the axis that makes fisher_tf sweepable. The
    exhaustion PRECEDES the entry (see evaluate_counter_trend), so the
    gate checks the recent window's extreme, not the entry-bar value.
    (0.0, 0.0) when history is insufficient."""
    if len(candles) < 12:
        return 0.0, 0.0
    line = fisher_transform(candles)[0][-window:]
    return min(line), max(line)


def simulate_counter_trend_outcome(candles: list[Candle], entry_index: int,
                                   signal: CounterTrendSignal,
                                   variant: str = "standard") -> TradeResult:
    """Walk bars after entry to a FIXED stop or the DYNAMIC opposite-cloud
    target (recomputed each bar as the Senkou spans move — this is why
    simulate_outcome, which assumes a fixed target, is not reused). Both
    touched in one bar -> stop first (conservative). No-lookahead: the
    target at bar j is the cloud computed from candles up to and including
    j only."""
    is_long = signal.direction == "LONG"
    risk = abs(signal.entry - signal.stop)
    entry_ts = datetime.fromtimestamp(candles[entry_index].close_time_ms / 1000, tz=timezone.utc)

    for j in range(entry_index + 1, len(candles)):
        c = candles[j]
        window = candles[max(0, j + 1 - CT_ICHIMOKU_WINDOW): j + 1]
        target = opposite_cloud_edge(window, is_long, variant=variant)
        hit_stop = c.low <= signal.stop if is_long else c.high >= signal.stop
        hit_target = target is not None and (c.high >= target if is_long else c.low <= target)
        if hit_stop or hit_target:
            exit_price = signal.stop if hit_stop else target   # stop wins ambiguity
            reason = "stop" if hit_stop else "target"
            sign = 1 if is_long else -1
            gross_r = sign * (exit_price - signal.entry) / risk
            fee_r = (signal.entry + exit_price) * TAKER_FEE / risk
            return TradeResult(
                entry_ts=entry_ts,
                exit_ts=datetime.fromtimestamp(c.close_time_ms / 1000, tz=timezone.utc),
                direction=signal.direction, entry=signal.entry, stop=signal.stop,
                target=signal.target_at_entry, reward_risk=signal.reward_risk,
                exit_reason=reason, gross_r=gross_r, net_r=gross_r - fee_r,
                bars_held=j - entry_index, indicators_snapshot={},
            )
    return TradeResult(entry_ts=entry_ts, exit_ts=None, direction=signal.direction,
                       entry=signal.entry, stop=signal.stop, target=signal.target_at_entry,
                       reward_risk=signal.reward_risk, exit_reason="unresolved",
                       gross_r=None, net_r=None, bars_held=len(candles) - 1 - entry_index,
                       indicators_snapshot={})


def run_counter_trend_backtest(bias_candles: list[Candle], trigger_candles: list[Candle],
                               fisher_from: str = "trigger", obv_rule: str = "divergence",
                               exhaustion_threshold: float = 2.0,
                               variant: str = "standard") -> dict:
    """Walk-forward over trigger closes for the E2E counter-trend module.
    fisher_from selects which series feeds the Fisher gate ('trigger' or
    'bias') — the fisher_tf sweep axis. One position at a time; no-
    lookahead bias slice, identical to the trend path."""
    if fisher_from not in ("trigger", "bias"):
        raise ValueError(f"fisher_from must be 'trigger' or 'bias', got {fisher_from!r}")
    trades: list[TradeResult] = []
    open_until_index = -1

    for i in range(WARMUP_TRIGGER_BARS, len(trigger_candles)):
        if i <= open_until_index:
            continue
        trig_slice = trigger_candles[max(0, i + 1 - LOOKBACK_BARS): i + 1]
        bias_all = bias_slice_no_lookahead(bias_candles, trigger_candles[i].close_time_ms)
        if len(bias_all) < WARMUP_BIAS_BARS:
            continue
        bias_slice = bias_all[-LOOKBACK_BARS:]
        fisher_slice = trig_slice if fisher_from == "trigger" else bias_slice
        fmin, fmax = _fisher_recent_extremes(fisher_slice, CT_EXHAUSTION_WINDOW)

        signal = evaluate_counter_trend(
            bias_slice, trig_slice, fmin, fmax, variant=variant,
            exhaustion_threshold=exhaustion_threshold, obv_rule=obv_rule,
        )
        if signal is None:
            continue
        trade = simulate_counter_trend_outcome(trigger_candles, i, signal, variant=variant)
        trade.indicators_snapshot = {
            "strategy": "counter_trend", "direction": signal.direction,
            "fisher_value": signal.fisher_value, "obv_rule": signal.obv_rule,
            "reason": signal.reason,
        }
        trades.append(trade)
        open_until_index = (i + trade.bars_held if trade.exit_ts is not None
                            else len(trigger_candles))

    return _summarize_trades(trades, len(trigger_candles) - WARMUP_TRIGGER_BARS)


def _cycle_leg_result(candles: list[Candle], entry_index: int, exit_index: int,
                      direction: str, entry: float, stop: float, exit_price: float,
                      reason: str, cycle_id: int) -> TradeResult:
    """One leg of a Fisher cycle. No fixed target/R:R (legs exit on the
    Fisher extreme, a stop, or a bias flip — not a price target), so those
    fields are None. Fees are charged on this leg's own entry+exit; a flip
    is a real close+reopen, so the shared price point is paid twice (once
    as this leg's exit, once as the next leg's entry)."""
    risk = abs(entry - stop)
    sign = 1 if direction == "LONG" else -1
    gross_r = sign * (exit_price - entry) / risk if risk else 0.0
    fee_r = (entry + exit_price) * TAKER_FEE / risk if risk else 0.0
    return TradeResult(
        entry_ts=datetime.fromtimestamp(candles[entry_index].close_time_ms / 1000, tz=timezone.utc),
        exit_ts=datetime.fromtimestamp(candles[exit_index].close_time_ms / 1000, tz=timezone.utc),
        direction=direction, entry=entry, stop=stop, target=None, reward_risk=None,
        exit_reason=reason, gross_r=gross_r, net_r=gross_r - fee_r,
        bars_held=exit_index - entry_index,
        indicators_snapshot={"strategy": "fisher_cycle", "cycle_id": cycle_id, "reason": reason},
    )


def run_fisher_cycle_backtest(bias_candles: list[Candle], trigger_candles: list[Candle],
                              exhaustion_threshold: float = 2.0,
                              atr_multiplier: float = 1.5) -> dict:
    """Track 3 multi-leg state machine, walked over 4H trigger bars with a
    1D structural bias (bias_candles). One leg at a time.

    Per bar: intrabar STOP first (stop-first ambiguity) → leg closes at the
    stop and goes FLAT (flat-and-rearm: a stop is the leg invalidated, NOT
    a reversal — earliest re-entry is the NEXT bar, no same-bar churn).
    Else at close: force-flatten if the 1D macro bias broke; otherwise flip
    LONG↔SHORT on a favorable Fisher exhaustion extreme (close + reopen
    opposite at this close). From FLAT, (re)open a pullback entry via
    opening_direction while the macro bias holds. No-lookahead: 1D bias
    from closed daily candles only; 4H Fisher + ATR precomputed causally."""
    fisher_line = fisher_transform(trigger_candles)[0]
    atr_series = wilder_atr(trigger_candles)

    legs: list[TradeResult] = []
    state = "FLAT"
    leg: dict | None = None          # {direction, entry, stop, entry_index}
    macro_dir = None                 # Bias the active cycle opened under; None = no cycle
    cycle_id = 0

    for i in range(WARMUP_TRIGGER_BARS, len(trigger_candles)):
        c = trigger_candles[i]
        fisher = fisher_line[i]
        atr = atr_series[i]
        bias = daily_bias_at(bias_candles, c.close_time_ms)
        stopped_this_bar = False

        if state != "FLAT":
            is_long = state == "LONG"
            hit_stop = c.low <= leg["stop"] if is_long else c.high >= leg["stop"]
            if hit_stop:
                legs.append(_cycle_leg_result(trigger_candles, leg["entry_index"], i,
                                              state, leg["entry"], leg["stop"], leg["stop"],
                                              "stop", cycle_id))
                state, leg, stopped_this_bar = "FLAT", None, True
                # flat-and-rearm: macro_dir stays; cycle continues, re-arm >= next bar
            elif macro_broken(macro_dir, bias):
                legs.append(_cycle_leg_result(trigger_candles, leg["entry_index"], i,
                                              state, leg["entry"], leg["stop"], c.close,
                                              "bias_flip", cycle_id))
                state, leg, macro_dir = "FLAT", None, None      # cycle ends
            elif is_exhausted(state, fisher, exhaustion_threshold):
                legs.append(_cycle_leg_result(trigger_candles, leg["entry_index"], i,
                                              state, leg["entry"], leg["stop"], c.close,
                                              "exhaustion_flip", cycle_id))
                new_dir = "SHORT" if is_long else "LONG"        # flip, same cycle
                leg = {"direction": new_dir, "entry": c.close,
                       "stop": leg_stop(new_dir, c.close, atr, atr_multiplier),
                       "entry_index": i}
                state = new_dir

        if state == "FLAT" and not stopped_this_bar and atr > 0.0:
            if macro_dir is not None and macro_broken(macro_dir, bias):
                macro_dir = None                                # stale cycle expired while flat
            the_bias = macro_dir if macro_dir is not None else bias
            direction = opening_direction(the_bias, fisher, exhaustion_threshold)
            if direction is not None:
                if macro_dir is None:                           # a fresh cycle begins
                    macro_dir = the_bias
                    cycle_id += 1
                leg = {"direction": direction, "entry": c.close,
                       "stop": leg_stop(direction, c.close, atr, atr_multiplier),
                       "entry_index": i}
                state = direction

    # flush a leg still open at data end as unresolved (excluded from
    # win/loss stats, same convention as the other simulators)
    if state != "FLAT" and leg is not None:
        last = len(trigger_candles) - 1
        legs.append(TradeResult(
            entry_ts=datetime.fromtimestamp(
                trigger_candles[leg["entry_index"]].close_time_ms / 1000, tz=timezone.utc),
            exit_ts=None, direction=state, entry=leg["entry"], stop=leg["stop"],
            target=None, reward_risk=None, exit_reason="unresolved",
            gross_r=None, net_r=None, bars_held=last - leg["entry_index"],
            indicators_snapshot={"strategy": "fisher_cycle", "cycle_id": cycle_id,
                                 "reason": "unresolved"}))

    summary = _summarize_trades(legs, len(trigger_candles) - WARMUP_TRIGGER_BARS)

    # per-cycle cumulative net R (the strategy's real performance unit —
    # a cycle is entry..bias-flip-flatten, spanning multiple legs)
    cycles: dict[int, dict] = {}
    for t in legs:
        cid = t.indicators_snapshot["cycle_id"]
        cyc = cycles.setdefault(cid, {"legs": 0, "resolved": 0, "net_r": 0.0})
        cyc["legs"] += 1
        if t.net_r is not None:
            cyc["resolved"] += 1
            cyc["net_r"] += t.net_r
    cycle_rs = [round(c["net_r"], 4) for c in cycles.values() if c["resolved"]]
    summary["cycles"] = {
        "count": len(cycles),
        "cumulative_r_per_cycle": cycle_rs,
        "mean_cycle_r": round(sum(cycle_rs) / len(cycle_rs), 4) if cycle_rs else None,
    }
    return summary


def _parse_indicators(spec: str) -> dict:
    if spec == "default":
        return dict(DEFAULT_INDICATOR_CONFIG)
    if spec == "all":
        return {n: True for n in INDICATOR_NAMES}
    chosen = {s.strip() for s in spec.split(",")}
    unknown = chosen - set(INDICATOR_NAMES)
    if unknown:
        raise SystemExit(f"unknown indicators: {unknown}")
    return {n: (n in chosen) for n in INDICATOR_NAMES}


CAVEATS = ("caveats: idealized touch fills, no slippage/funding, stop-first on "
           "ambiguous candles, taker fees 0.075%/side, window limited to most "
           "recent 5,000 trigger candles.")


def _window(trigger_candles: list[Candle]) -> tuple[datetime, datetime]:
    t0 = datetime.fromtimestamp(trigger_candles[0].open_time_ms / 1000, tz=timezone.utc)
    t1 = datetime.fromtimestamp(trigger_candles[-1].close_time_ms / 1000, tz=timezone.utc)
    return t0, t1


def store_run(conn, run_id: str, bias_tf: str, trigger_tf: str, config: dict,
              t0: datetime, t1: datetime, summary: dict,
              trades: list[TradeResult], notes: dict,
              strategy_type: str = "trend") -> None:
    conn.execute(
        """INSERT INTO backtest_runs (run_id, bias_tf, trigger_tf, indicator_config,
               candles_from, candles_to, bars_evaluated, trades, wins, losses,
               unresolved, suppressed_rr, gross_r, net_r, avg_net_r, win_rate,
               profit_factor, max_drawdown_r, fees_model, notes, strategy_type)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (run_id, bias_tf, trigger_tf, json.dumps(config), t0, t1,
         summary["bars_evaluated"], len(trades), summary["wins"], summary["losses"],
         summary["unresolved"], summary["suppressed_rr"], summary["gross_r"],
         summary["net_r"], summary["avg_net_r"], summary["win_rate"],
         summary["profit_factor"], summary["max_drawdown_r"],
         "taker 0.075%/side, no slippage/funding",
         json.dumps(notes, default=str), strategy_type),
    )
    for t in trades:
        conn.execute(
            """INSERT INTO backtest_trades (run_id, entry_ts, exit_ts, direction,
                   entry, stop, target, reward_risk, exit_reason, gross_r, net_r,
                   bars_held, indicators_snapshot)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id, t.entry_ts, t.exit_ts, t.direction, t.entry, t.stop, t.target,
             t.reward_risk, t.exit_reason, t.gross_r, t.net_r, t.bars_held,
             json.dumps(t.indicators_snapshot, default=str)),
        )


def _summary_row(summary: dict) -> str:
    wr = f"{summary['win_rate']:.0%}" if summary["win_rate"] is not None else "-"
    pf = f"{summary['profit_factor']:.2f}" if summary["profit_factor"] is not None else "-"
    return (f"trades {len(summary['trades']):>3} | "
            f"W-L {summary['wins']}-{summary['losses']} | wr {wr:>4} | "
            f"net {summary['net_r']:+7.2f}R | PF {pf:>5} | "
            f"maxDD {summary['max_drawdown_r']:5.2f}R | "
            f"supp_rr {summary['suppressed_rr']}"
            + (f" | supp_std {summary['suppressed_standdown']}"
               if summary.get("suppressed_standdown") else ""))


# ── single-run mode ──

def run_single(args) -> None:
    config = _parse_indicators(args.indicators)
    now_ms = int(time.time() * 1000)
    span = lambda tf: 5000 * interval_seconds(tf) * 1000

    print(f"fetching history: bias {args.bias_tf}, trigger {args.trigger_tf} ...")
    bias_candles = fetch_candles("BTC", args.bias_tf, now_ms - span(args.bias_tf), now_ms)
    trigger_candles = fetch_candles("BTC", args.trigger_tf, now_ms - span(args.trigger_tf), now_ms)
    t0, t1 = _window(trigger_candles)
    print(f"  {len(bias_candles)} bias candles, {len(trigger_candles)} trigger candles "
          f"({t0:%Y-%m-%d} -> {t1:%Y-%m-%d})")

    candles_4h = None
    if args.fisher4h_entry or args.fisher4h_exit:
        candles_4h = (bias_candles if args.bias_tf == "4h"
                      else trigger_candles if args.trigger_tf == "4h"
                      else fetch_candles("BTC", "4h", now_ms - span("4h"), now_ms))

    funding_series = load_funding_series(now_ms) if args.standdown_entry else None
    oi_series = load_oi_series() if (args.standdown_entry and args.oi_z_min is not None) else None

    summary = run_backtest(bias_candles, trigger_candles, config, args.ichimoku_variant,
                           stop_model=args.stop_model, atr_multiplier=args.atr_multiplier,
                           target_model=args.target_model,
                           blue_sky_atr_multiplier=args.blue_sky_atr_multiplier,
                           fisher4h_entry=args.fisher4h_entry, fisher4h_exit=args.fisher4h_exit,
                           exhaustion_threshold=args.exhaustion_threshold,
                           candles_4h=candles_4h,
                           standdown_entry=args.standdown_entry,
                           funding_series=funding_series,
                           funding_pctile_threshold=args.funding_pctile,
                           oi_series=oi_series,
                           oi_z_min=args.oi_z_min)
    trades = summary.pop("trades")

    print("\n=== SIMULATED BACKTEST RESULT (not live performance) ===")
    print(f"combo: {args.bias_tf} bias / {args.trigger_tf} trigger | indicators: "
          + "+".join(n for n, v in config.items() if v)
          + f" | stop: {args.stop_model}"
          + (f"@{args.atr_multiplier}" if args.stop_model == "hybrid" else "")
          + f" | target: {args.target_model}"
          + (f"@{args.blue_sky_atr_multiplier}" if args.target_model == "blue_sky_atr" else "")
          + (f" | fisher4h E={args.fisher4h_entry} X={args.fisher4h_exit}"
             f"@{args.exhaustion_threshold}"
             if (args.fisher4h_entry or args.fisher4h_exit) else "")
          + (f" | standdown fund>={args.funding_pctile}pct"
             + (f" & OIz>={args.oi_z_min}" if args.oi_z_min is not None else " (OI dormant)")
             if args.standdown_entry else ""))
    print(f"window: {t0:%Y-%m-%d} -> {t1:%Y-%m-%d} | bars evaluated: {summary['bars_evaluated']}")
    print(f"signals taken: {len(trades)} (resolved {summary['resolved']}, "
          f"unresolved {summary['unresolved']}) | suppressed by R:R gate: {summary['suppressed_rr']}"
          + (f" | by 4H exhaustion: {summary['suppressed_exhaustion']}"
             if summary["suppressed_exhaustion"] else "")
          + (f" | by stand-down: {summary['suppressed_standdown']}"
             if summary["suppressed_standdown"] else ""))
    if summary["resolved"]:
        print(f"wins {summary['wins']} / losses {summary['losses']} "
              f"(win rate {summary['win_rate']:.1%})")
        print(f"net {summary['net_r']:+.2f}R total | avg {summary['avg_net_r']:+.3f}R/trade | "
              f"profit factor {summary['profit_factor'] and round(summary['profit_factor'], 2)} | "
              f"max drawdown {summary['max_drawdown_r']:.2f}R")
    else:
        print("no resolved trades in window -- insufficient data, no conclusions")
    print(CAVEATS)

    if not args.no_store:
        from db.store import TelemetryStore
        store = TelemetryStore()
        run_id = str(ULID())
        notes = {
            "kind": "SIMULATED walk-forward via live strategy code",
            "stop_model": args.stop_model,
            "atr_multiplier": args.atr_multiplier if args.stop_model == "hybrid" else None,
            "target_model": args.target_model,
            "blue_sky_atr_multiplier": (args.blue_sky_atr_multiplier
                                        if args.target_model == "blue_sky_atr" else None),
            "fisher4h_entry": args.fisher4h_entry,
            "fisher4h_exit": args.fisher4h_exit,
            "exhaustion_threshold": (args.exhaustion_threshold
                                     if (args.fisher4h_entry or args.fisher4h_exit) else None),
            "suppressed_exhaustion": summary["suppressed_exhaustion"],
            "standdown_entry": args.standdown_entry,
            "funding_pctile_threshold": args.funding_pctile if args.standdown_entry else None,
            "oi_z_min": args.oi_z_min if args.standdown_entry else None,
            "oi_used": False,
            "suppressed_standdown": summary["suppressed_standdown"],
            "return_stats": {args.bias_tf: log_return_stats(bias_candles),
                             args.trigger_tf: log_return_stats(trigger_candles)},
        }
        store_run(store._connect(), run_id, args.bias_tf, args.trigger_tf, config,
                  t0, t1, summary, trades, notes)
        print(f"stored: run_id={run_id} ({len(trades)} trades) in backtest_runs/backtest_trades")


# ── sweep mode ──

def expand_sweep(cfg: dict) -> list[dict]:
    """Cross-product expansion per grid. Config-driven so future sweeps
    are a YAML edit, not a code change. A grid without a fisher4h key
    runs with both exhaustion mechanisms off; entries with either
    mechanism on expand once per listed threshold."""
    combos: list[dict] = []
    for grid in cfg["grids"]:
        fisher_variants = grid.get("fisher4h", [{"entry": False, "exit": False}])
        standdown_variants = grid.get("standdown", [{"entry": False}])
        target_models = grid.get("target_models", ["nearest_structure"])
        unknown_targets = set(target_models) - set(TARGET_MODELS)
        if unknown_targets:
            raise SystemExit(f"unknown target_models: {unknown_targets}")
        blue_sky_mult = grid.get("blue_sky_atr_multiplier", DEFAULT_BLUE_SKY_ATR_MULTIPLIER)
        for tf in grid["tf_pairs"]:
            validate_combo(tf["bias"], tf["trigger"])
            for ind in grid["indicator_sets"]:
                for sm in grid["stop_models"]:
                    if sm["model"] == "hybrid":
                        mult = sm.get("atr_multiplier")
                        if mult is None:
                            raise SystemExit("hybrid stop_model entries need atr_multiplier")
                    else:
                        mult = None
                    for tm in target_models:
                        for fv in fisher_variants:
                            entry, exit_ = bool(fv.get("entry")), bool(fv.get("exit"))
                            thresholds = fv.get("thresholds", [FISHER4H_EXHAUSTION_THRESHOLD]) \
                                if (entry or exit_) else [None]
                            for thr in thresholds:
                                for sd in standdown_variants:
                                    sd_on = bool(sd.get("entry"))
                                    pctiles = sd.get("funding_pctiles", [85.0]) if sd_on else [None]
                                    z_mins = sd.get("oi_z_mins", [None]) if sd_on else [None]
                                    for pct in pctiles:
                                        for zm in z_mins:
                                            combos.append({
                                                "grid": grid["name"],
                                                "bias_tf": tf["bias"], "trigger_tf": tf["trigger"],
                                                "indicators": ind,
                                                "stop_model": sm["model"], "atr_multiplier": mult,
                                                "target_model": tm,
                                                "blue_sky_atr_multiplier": (
                                                    blue_sky_mult if tm == "blue_sky_atr" else None),
                                                "fisher4h_entry": entry, "fisher4h_exit": exit_,
                                                "exhaustion_threshold": thr,
                                                "standdown_entry": sd_on,
                                                "funding_pctile": pct,
                                                "oi_z_min": zm,
                                            })
    return combos


def _fisher_label(c: dict) -> str:
    if not (c["fisher4h_entry"] or c["fisher4h_exit"]):
        return "off"
    parts = ("E" if c["fisher4h_entry"] else "") + ("X" if c["fisher4h_exit"] else "")
    return f"{parts}@{c['exhaustion_threshold']}"


def _standdown_label(c: dict) -> str:
    if not c.get("standdown_entry"):
        return "off"
    label = f"F{c['funding_pctile']:.0f}"
    if c.get("oi_z_min") is not None:
        label += f"+Z{c['oi_z_min']}"
    return label


_TARGET_ABBREV = {"nearest_structure": "nearest", "fib_extension_preferred": "fib_ext",
                  "blue_sky_atr": "blue_sky"}


def _target_label(c: dict) -> str:
    label = _TARGET_ABBREV[c["target_model"]]
    if c["blue_sky_atr_multiplier"]:
        label += f"@{c['blue_sky_atr_multiplier']}"
    return label


def _combo_label(c: dict) -> str:
    stop = c["stop_model"] + (f"@{c['atr_multiplier']}" if c["atr_multiplier"] else "")
    return (f"{c['bias_tf']}/{c['trigger_tf']} | {c['indicators']:<24} | "
            f"{stop:<14} | tgt {_target_label(c):<12} | f4h {_fisher_label(c):<7} | "
            f"std {_standdown_label(c):<8}")


def run_sweep(args) -> None:
    with open(args.sweep, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    combos = expand_sweep(cfg)
    sweep_id = str(ULID())
    print(f"sweep {cfg.get('sweep_name', 'unnamed')} | {len(combos)} runs | sweep_id={sweep_id}")

    # fetch each unique TF ONCE — consistent windows across every run.
    # 4h is always included when any combo runs a fisher4h variant (the
    # exhaustion heuristic reads the 4H chart regardless of the TF pair).
    now_ms = int(time.time() * 1000)
    span = lambda tf: 5000 * interval_seconds(tf) * 1000
    unique_tfs = {c["bias_tf"] for c in combos} | {c["trigger_tf"] for c in combos}
    if any(c["fisher4h_entry"] or c["fisher4h_exit"] for c in combos):
        unique_tfs.add("4h")
    funding_series = (load_funding_series(now_ms)
                      if any(c.get("standdown_entry") for c in combos) else None)
    if funding_series:
        from datetime import datetime, timezone
        f0 = datetime.fromtimestamp(funding_series[0][0] / 1000, tz=timezone.utc)
        f1 = datetime.fromtimestamp(funding_series[-1][0] / 1000, tz=timezone.utc)
        print(f"  funding series: {len(funding_series)} hourly rows ({f0:%Y-%m-%d} -> {f1:%Y-%m-%d})")
    oi_series = (load_oi_series()
                 if any(c.get("standdown_entry") and c.get("oi_z_min") is not None for c in combos)
                 else None)
    if oi_series:
        from datetime import datetime, timezone
        o0 = datetime.fromtimestamp(oi_series[0][0] / 1000, tz=timezone.utc)
        o1 = datetime.fromtimestamp(oi_series[-1][0] / 1000, tz=timezone.utc)
        print(f"  OI series: {len(oi_series)} hourly rows ({o0:%Y-%m-%d} -> {o1:%Y-%m-%d}) "
              f"— conjunction declines to fire on bars before {o0:%Y-%m-%d}+30d")
    unique_tfs = sorted(unique_tfs, key=interval_seconds)
    candles: dict[str, list[Candle]] = {}
    for tf in unique_tfs:
        candles[tf] = fetch_candles("BTC", tf, now_ms - span(tf), now_ms)
        t0, t1 = _window(candles[tf])
        print(f"  fetched {tf}: {len(candles[tf])} candles ({t0:%Y-%m-%d} -> {t1:%Y-%m-%d})")
    stats = {tf: log_return_stats(candles[tf]) for tf in unique_tfs}

    store_conn = None
    if not args.no_store:
        from db.store import TelemetryStore
        store_conn = TelemetryStore()._connect()

    results: list[tuple[dict, dict, str]] = []
    for idx, combo in enumerate(combos, 1):
        config = _parse_indicators(combo["indicators"])
        summary = run_backtest(
            candles[combo["bias_tf"]], candles[combo["trigger_tf"]], config,
            stop_model=combo["stop_model"],
            atr_multiplier=combo["atr_multiplier"] or DEFAULT_ATR_MULTIPLIER,
            target_model=combo["target_model"],
            blue_sky_atr_multiplier=(combo["blue_sky_atr_multiplier"]
                                     or DEFAULT_BLUE_SKY_ATR_MULTIPLIER),
            fisher4h_entry=combo["fisher4h_entry"],
            fisher4h_exit=combo["fisher4h_exit"],
            exhaustion_threshold=combo["exhaustion_threshold"] or FISHER4H_EXHAUSTION_THRESHOLD,
            candles_4h=candles.get("4h"),
            standdown_entry=combo["standdown_entry"],
            funding_series=funding_series if combo["standdown_entry"] else None,
            funding_pctile_threshold=combo["funding_pctile"] or 85.0,
            oi_series=oi_series if combo["oi_z_min"] is not None else None,
            oi_z_min=combo["oi_z_min"],
        )
        trades = summary["trades"]
        run_id = str(ULID())
        if store_conn is not None:
            t0, t1 = _window(candles[combo["trigger_tf"]])
            notes = {
                "kind": "SIMULATED walk-forward via live strategy code",
                "sweep_id": sweep_id, "grid": combo["grid"],
                "stop_model": combo["stop_model"],
                "atr_multiplier": combo["atr_multiplier"],
                "target_model": combo["target_model"],
                "blue_sky_atr_multiplier": combo["blue_sky_atr_multiplier"],
                "fisher4h_entry": combo["fisher4h_entry"],
                "fisher4h_exit": combo["fisher4h_exit"],
                "exhaustion_threshold": combo["exhaustion_threshold"],
                "suppressed_exhaustion": summary["suppressed_exhaustion"],
                "standdown_entry": combo["standdown_entry"],
                "funding_pctile_threshold": combo["funding_pctile"],
                "oi_z_min": combo["oi_z_min"],
                "oi_used": bool(combo["oi_z_min"]),
                "suppressed_standdown": summary["suppressed_standdown"],
                "return_stats": {combo["bias_tf"]: stats[combo["bias_tf"]],
                                 combo["trigger_tf"]: stats[combo["trigger_tf"]]},
            }
            store_run(store_conn, run_id, combo["bias_tf"], combo["trigger_tf"], config,
                      t0, t1, summary, trades, notes)
        print(f"[{idx:>3}/{len(combos)}] {_combo_label(combo)} | {_summary_row(summary)}")
        results.append((combo, summary, run_id))

    print("\n=== SIMULATED SWEEP COMPARISON (not live performance) ===")
    print(f"sweep_id={sweep_id} | runs={len(results)}")
    header = (f"{'grid':<14} {'tfs':<9} {'indicators':<24} {'stop':<14} {'target':<13} "
              f"{'f4h':<8} {'std':<9} {'trades':>6} {'W-L':>7} {'netR':>8} {'PF':>6} {'maxDD':>7} "
              f"{'supp_rr':>8} {'supp_exh':>9} {'supp_std':>9}")
    print(header)
    print("-" * len(header))
    for combo, summary, _ in results:
        pf = summary["profit_factor"]
        stop = combo["stop_model"] + (f"@{combo['atr_multiplier']}" if combo["atr_multiplier"] else "")
        print(f"{combo['grid']:<14} {combo['bias_tf'] + '/' + combo['trigger_tf']:<9} "
              f"{combo['indicators']:<24} {stop:<14} {_target_label(combo):<13} "
              f"{_fisher_label(combo):<8} {_standdown_label(combo):<9} "
              f"{len(summary['trades']):>6} {str(summary['wins']) + '-' + str(summary['losses']):>7} "
              f"{summary['net_r']:>+8.2f} {(f'{pf:.2f}' if pf is not None else '-'):>6} "
              f"{summary['max_drawdown_r']:>7.2f} {summary['suppressed_rr']:>8} "
              f"{summary['suppressed_exhaustion']:>9} {summary['suppressed_standdown']:>9}")
    print(CAVEATS)
    if store_conn is not None:
        print(f"stored: {len(results)} runs under sweep_id={sweep_id} in backtest_runs/backtest_trades")


def _fisher_from(fisher_tf: str, bias_tf: str, trigger_tf: str) -> str:
    if fisher_tf == trigger_tf:
        return "trigger"
    if fisher_tf == bias_tf:
        return "bias"
    raise SystemExit(f"fisher-tf {fisher_tf} must equal bias-tf ({bias_tf}) or "
                     f"trigger-tf ({trigger_tf}) — those are the fetched series")


def run_counter_trend_single(args) -> None:
    now_ms = int(time.time() * 1000)
    span = lambda tf: 5000 * interval_seconds(tf) * 1000
    print(f"fetching history: bias {args.bias_tf}, trigger {args.trigger_tf} ... (counter_trend)")
    bias_candles = fetch_candles("BTC", args.bias_tf, now_ms - span(args.bias_tf), now_ms)
    trigger_candles = fetch_candles("BTC", args.trigger_tf, now_ms - span(args.trigger_tf), now_ms)
    t0, t1 = _window(trigger_candles)
    fisher_tf = args.fisher_tf or args.trigger_tf
    fisher_from = _fisher_from(fisher_tf, args.bias_tf, args.trigger_tf)

    summary = run_counter_trend_backtest(
        bias_candles, trigger_candles, fisher_from=fisher_from,
        obv_rule=args.obv_rule, exhaustion_threshold=args.exhaustion_threshold)
    trades = summary.pop("trades")

    print("\n=== SIMULATED COUNTER-TREND RESULT (not live performance) ===")
    print(f"combo: {args.bias_tf} bias / {args.trigger_tf} trigger | fisher_tf {fisher_tf} | "
          f"obv {args.obv_rule} | exhaustion {args.exhaustion_threshold}")
    print(f"window: {t0:%Y-%m-%d} -> {t1:%Y-%m-%d} | bars evaluated: {summary['bars_evaluated']}")
    print(f"signals taken: {len(trades)} (resolved {summary['resolved']}, "
          f"unresolved {summary['unresolved']})")
    if summary["resolved"]:
        print(f"wins {summary['wins']} / losses {summary['losses']} "
              f"(win rate {summary['win_rate']:.1%})")
        print(f"net {summary['net_r']:+.2f}R total | avg {summary['avg_net_r']:+.3f}R/trade | "
              f"profit factor {summary['profit_factor'] and round(summary['profit_factor'], 2)} | "
              f"max drawdown {summary['max_drawdown_r']:.2f}R")
    else:
        print("no resolved trades in window -- insufficient data, no conclusions")
    print(CAVEATS)

    if not args.no_store:
        from db.store import TelemetryStore
        store = TelemetryStore()
        run_id = str(ULID())
        notes = {
            "kind": "SIMULATED counter-trend (Ichimoku E2E + Fisher + OBV)",
            "strategy": "counter_trend", "fisher_tf": fisher_tf,
            "obv_rule": args.obv_rule, "exhaustion_threshold": args.exhaustion_threshold,
            "return_stats": {args.bias_tf: log_return_stats(bias_candles),
                             args.trigger_tf: log_return_stats(trigger_candles)},
        }
        store_run(store._connect(), run_id, args.bias_tf, args.trigger_tf,
                  {"strategy": "counter_trend"}, t0, t1, summary, trades, notes,
                  strategy_type="counter_trend")
        print(f"stored: run_id={run_id} ({len(trades)} trades, counter_trend)")


def run_counter_trend_sweep(cfg: dict, args) -> None:
    tf = cfg.get("tf_pair", {"bias": "4h", "trigger": "1h"})
    bias_tf, trigger_tf = tf["bias"], tf["trigger"]
    validate_combo(bias_tf, trigger_tf)
    fisher_tfs = cfg.get("fisher_tfs", [trigger_tf, bias_tf])
    obv_rules = cfg.get("obv_rules", list(OBV_RULES))
    thresholds = cfg.get("exhaustion_thresholds", [1.5, 2.0, 2.5])
    unknown = set(obv_rules) - set(OBV_RULES)
    if unknown:
        raise SystemExit(f"unknown obv_rules: {unknown}")

    combos = [(ft, obv, thr) for ft in fisher_tfs for obv in obv_rules for thr in thresholds]
    sweep_id = str(ULID())
    print(f"sweep {cfg.get('sweep_name', 'counter_trend')} | {len(combos)} runs | "
          f"sweep_id={sweep_id}")

    now_ms = int(time.time() * 1000)
    span = lambda t: 5000 * interval_seconds(t) * 1000
    bias_candles = fetch_candles("BTC", bias_tf, now_ms - span(bias_tf), now_ms)
    trigger_candles = fetch_candles("BTC", trigger_tf, now_ms - span(trigger_tf), now_ms)
    t0, t1 = _window(trigger_candles)
    print(f"  fetched {bias_tf}: {len(bias_candles)} | {trigger_tf}: {len(trigger_candles)} "
          f"({t0:%Y-%m-%d} -> {t1:%Y-%m-%d})")
    stats = {bias_tf: log_return_stats(bias_candles), trigger_tf: log_return_stats(trigger_candles)}

    store_conn = None
    if not args.no_store:
        from db.store import TelemetryStore
        store_conn = TelemetryStore()._connect()

    results: list[tuple[tuple, dict, str]] = []
    for idx, (ft, obv, thr) in enumerate(combos, 1):
        summary = run_counter_trend_backtest(
            bias_candles, trigger_candles,
            fisher_from=_fisher_from(ft, bias_tf, trigger_tf),
            obv_rule=obv, exhaustion_threshold=thr)
        trades = summary["trades"]
        run_id = str(ULID())
        if store_conn is not None:
            notes = {
                "kind": "SIMULATED counter-trend (Ichimoku E2E + Fisher + OBV)",
                "sweep_id": sweep_id, "strategy": "counter_trend",
                "fisher_tf": ft, "obv_rule": obv, "exhaustion_threshold": thr,
                "return_stats": stats,
            }
            store_run(store_conn, run_id, bias_tf, trigger_tf, {"strategy": "counter_trend"},
                      t0, t1, summary, trades, notes, strategy_type="counter_trend")
        print(f"[{idx:>2}/{len(combos)}] fisher {ft:<3} | obv {obv:<14} | exh {thr} | "
              f"{_summary_row(summary)}")
        results.append(((ft, obv, thr), summary, run_id))

    print("\n=== SIMULATED COUNTER-TREND SWEEP COMPARISON (not live performance) ===")
    print(f"sweep_id={sweep_id} | runs={len(results)} | {bias_tf}/{trigger_tf}")
    header = (f"{'fisher_tf':<10}{'obv_rule':<16}{'exh':<6}{'trades':>7}{'W-L':>8}"
              f"{'netR':>9}{'PF':>7}{'maxDD':>8}")
    print(header)
    print("-" * len(header))
    for (ft, obv, thr), summary, _ in results:
        pf = summary["profit_factor"]
        print(f"{ft:<10}{obv:<16}{str(thr):<6}{len(summary['trades']):>7}"
              f"{str(summary['wins']) + '-' + str(summary['losses']):>8}"
              f"{summary['net_r']:>+9.2f}{(f'{pf:.2f}' if pf is not None else '-'):>7}"
              f"{summary['max_drawdown_r']:>8.2f}")
    print(CAVEATS)
    if store_conn is not None:
        print(f"stored: {len(results)} runs under sweep_id={sweep_id} (strategy_type=counter_trend)")


def run_fisher_cycle_single(args) -> None:
    bias_tf = args.bias_tf if args.bias_tf != "4h" else "1d"   # cycle default is 1D bias / 4H trigger
    trigger_tf = args.trigger_tf if args.trigger_tf != "1h" else "4h"
    now_ms = int(time.time() * 1000)
    span = lambda t: 5000 * interval_seconds(t) * 1000
    print(f"fetching history: bias {bias_tf}, trigger {trigger_tf} ... (fisher_cycle)")
    bias_candles = fetch_candles("BTC", bias_tf, now_ms - span(bias_tf), now_ms)
    trigger_candles = fetch_candles("BTC", trigger_tf, now_ms - span(trigger_tf), now_ms)
    t0, t1 = _window(trigger_candles)
    summary = run_fisher_cycle_backtest(bias_candles, trigger_candles,
                                        exhaustion_threshold=args.exhaustion_threshold,
                                        atr_multiplier=args.atr_multiplier)
    trades, cycles = summary.pop("trades"), summary["cycles"]
    print("\n=== SIMULATED FISHER-CYCLE RESULT (not live performance) ===")
    print(f"combo: {bias_tf} bias / {trigger_tf} trigger | exhaustion {args.exhaustion_threshold} "
          f"| atr_mult {args.atr_multiplier}")
    print(f"window: {t0:%Y-%m-%d} -> {t1:%Y-%m-%d} | bars evaluated: {summary['bars_evaluated']}")
    print(f"cycles {cycles['count']} | legs {len(trades)} (resolved {summary['resolved']}, "
          f"unresolved {summary['unresolved']})")
    if summary["resolved"]:
        print(f"leg wins {summary['wins']} / losses {summary['losses']} "
              f"(win rate {summary['win_rate']:.1%})")
        print(f"net {summary['net_r']:+.2f}R total | profit factor "
              f"{summary['profit_factor'] and round(summary['profit_factor'], 2)} | "
              f"max drawdown {summary['max_drawdown_r']:.2f}R | mean cycle R {cycles['mean_cycle_r']}")
    else:
        print("no resolved legs in window -- insufficient data, no conclusions")
    print(CAVEATS)
    if not args.no_store:
        from db.store import TelemetryStore
        run_id = str(ULID())
        notes = {"kind": "SIMULATED fisher-cycle (1D bias + 4H Fisher pullback/exhaustion cycling)",
                 "strategy": "fisher_cycle", "exhaustion_threshold": args.exhaustion_threshold,
                 "atr_multiplier": args.atr_multiplier, "cycles": cycles,
                 "return_stats": {bias_tf: log_return_stats(bias_candles),
                                  trigger_tf: log_return_stats(trigger_candles)}}
        store_run(TelemetryStore()._connect(), run_id, bias_tf, trigger_tf,
                  {"strategy": "fisher_cycle"}, t0, t1, summary, trades, notes,
                  strategy_type="fisher_cycle")
        print(f"stored: run_id={run_id} ({len(trades)} legs, fisher_cycle)")


def run_fisher_cycle_sweep(cfg: dict, args) -> None:
    tf = cfg.get("tf_pair", {"bias": "1d", "trigger": "4h"})
    bias_tf, trigger_tf = tf["bias"], tf["trigger"]
    validate_combo(bias_tf, trigger_tf)
    thresholds = cfg.get("exhaustion_thresholds", [1.5, 2.0, 2.5])
    atr_mults = cfg.get("atr_multipliers", [1.0, 1.5])
    combos = [(thr, am) for thr in thresholds for am in atr_mults]
    sweep_id = str(ULID())
    print(f"sweep {cfg.get('sweep_name', 'fisher_cycle')} | {len(combos)} runs | sweep_id={sweep_id}")

    now_ms = int(time.time() * 1000)
    span = lambda t: 5000 * interval_seconds(t) * 1000
    bias_candles = fetch_candles("BTC", bias_tf, now_ms - span(bias_tf), now_ms)
    trigger_candles = fetch_candles("BTC", trigger_tf, now_ms - span(trigger_tf), now_ms)
    t0, t1 = _window(trigger_candles)
    print(f"  fetched {bias_tf}: {len(bias_candles)} | {trigger_tf}: {len(trigger_candles)} "
          f"({t0:%Y-%m-%d} -> {t1:%Y-%m-%d})")
    stats = {bias_tf: log_return_stats(bias_candles), trigger_tf: log_return_stats(trigger_candles)}

    store_conn = None
    if not args.no_store:
        from db.store import TelemetryStore
        store_conn = TelemetryStore()._connect()

    results: list[tuple[tuple, dict, str]] = []
    for idx, (thr, am) in enumerate(combos, 1):
        summary = run_fisher_cycle_backtest(bias_candles, trigger_candles,
                                            exhaustion_threshold=thr, atr_multiplier=am)
        trades, cycles = summary["trades"], summary["cycles"]
        run_id = str(ULID())
        if store_conn is not None:
            notes = {
                "kind": "SIMULATED fisher-cycle (1D bias + 4H Fisher pullback/exhaustion cycling)",
                "sweep_id": sweep_id, "strategy": "fisher_cycle",
                "exhaustion_threshold": thr, "atr_multiplier": am,
                "cycles": cycles, "return_stats": stats,
            }
            store_run(store_conn, run_id, bias_tf, trigger_tf, {"strategy": "fisher_cycle"},
                      t0, t1, summary, trades, notes, strategy_type="fisher_cycle")
        print(f"[{idx}/{len(combos)}] exh {thr} | atr {am} | cycles {cycles['count']} | "
              f"{_summary_row(summary)}")
        results.append(((thr, am), summary, run_id))

    print("\n=== SIMULATED FISHER-CYCLE SWEEP COMPARISON (not live performance) ===")
    print(f"sweep_id={sweep_id} | runs={len(results)} | {bias_tf}/{trigger_tf}")
    header = (f"{'exh':<6}{'atr':<6}{'cycles':>8}{'legs':>7}{'W-L':>8}{'netR':>9}"
              f"{'PF':>7}{'maxDD':>8}{'meanCycleR':>12}")
    print(header)
    print("-" * len(header))
    for (thr, am), summary, _ in results:
        pf = summary["profit_factor"]
        mcr = summary["cycles"]["mean_cycle_r"]
        print(f"{str(thr):<6}{str(am):<6}{summary['cycles']['count']:>8}"
              f"{len(summary['trades']):>7}"
              f"{str(summary['wins']) + '-' + str(summary['losses']):>8}"
              f"{summary['net_r']:>+9.2f}{(f'{pf:.2f}' if pf is not None else '-'):>7}"
              f"{summary['max_drawdown_r']:>8.2f}{(f'{mcr:+.3f}' if mcr is not None else '-'):>12}")
    print("legs = individual long/short legs; cycles = entry..bias-flip macro runs "
          "(performance unit). No per-leg R:R gate by design.")
    print(CAVEATS)
    if store_conn is not None:
        print(f"stored: {len(results)} runs under sweep_id={sweep_id} (strategy_type=fisher_cycle)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", help="YAML sweep config; runs a batch instead of a single backtest")
    ap.add_argument("--strategy", default="trend",
                    choices=("trend", "counter_trend", "fisher_cycle"))
    ap.add_argument("--bias-tf", default="4h")
    ap.add_argument("--trigger-tf", default="1h")
    ap.add_argument("--indicators", default="default")
    ap.add_argument("--ichimoku-variant", default="standard")
    ap.add_argument("--stop-model", default="structural", choices=("structural", "hybrid"))
    ap.add_argument("--atr-multiplier", type=float, default=DEFAULT_ATR_MULTIPLIER)
    ap.add_argument("--target-model", default="nearest_structure", choices=TARGET_MODELS)
    ap.add_argument("--blue-sky-atr-multiplier", type=float,
                    default=DEFAULT_BLUE_SKY_ATR_MULTIPLIER)
    ap.add_argument("--fisher4h-entry", action="store_true",
                    help="suppress entries when 4H Fisher already extended in signal direction")
    ap.add_argument("--fisher4h-exit", action="store_true",
                    help="exit open sim positions when 4H Fisher crosses extended in trade's favor")
    ap.add_argument("--exhaustion-threshold", type=float, default=FISHER4H_EXHAUSTION_THRESHOLD)
    ap.add_argument("--standdown-entry", action="store_true",
                    help="suppress entries in the crowded direction when funding is at a "
                         "trailing-30d percentile extreme (OI z conjunction when --oi-z-min set)")
    ap.add_argument("--funding-pctile", type=float, default=85.0)
    ap.add_argument("--oi-z-min", type=float, default=None)
    ap.add_argument("--fisher-tf", default=None,
                    help="counter_trend: series for the Fisher gate (must equal bias-tf or trigger-tf)")
    ap.add_argument("--obv-rule", default="divergence", choices=OBV_RULES)
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        with open(args.sweep, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        strategy = cfg.get("strategy", "trend")
        if strategy == "counter_trend":
            run_counter_trend_sweep(cfg, args)
        elif strategy == "fisher_cycle":
            run_fisher_cycle_sweep(cfg, args)
        else:
            run_sweep(args)
    elif args.strategy == "counter_trend":
        run_counter_trend_single(args)
    elif args.strategy == "fisher_cycle":
        run_fisher_cycle_single(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
