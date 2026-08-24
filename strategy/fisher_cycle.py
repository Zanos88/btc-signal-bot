"""Track 3 — 1D-bias + 4H-Fisher pullback-entry & exhaustion-cycling.

ISOLATED, BACKTEST-ONLY: no import from strategy/signals.py, never wired
into the live/dry-run engine. A multi-leg state machine (not discrete
R:R-gated bets), driven by the outcome simulator in backtest.py.

Thesis: inside a favorable 1D structural bias, a 4H Fisher extreme
AGAINST the immediate move is a pullback entry (buy the dip in a bullish
1D bias). Once in, an opposite Fisher extreme (exhaustion in the trade's
favor) FLIPS the leg to bank the pullback, cycling while the 1D bias
holds. Every leg carries its own ATR hard stop — the failure mode
(a trend pinning Fisher extended so the reset never comes) is capped by
the stop, not left open.

Stop semantics (user decision): a stop-out is the leg being invalidated,
NOT a reversal signal — so a stopped leg goes FLAT and the cycle re-arms
via the normal pullback-entry condition while the macro bias holds. Only
the Fisher EXHAUSTION extreme flips leg direction, never a stop.

Part 1 here is 1D bias access: compute_bias (strategy/bias_4h) is already
timeframe-agnostic (it consumes any candle sequence), so 1D bias needs no
new logic — just a no-lookahead daily slice at the current 4H timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from data.feed import Candle
from strategy.bias_4h import Bias, compute_bias

DEFAULT_EXHAUSTION_THRESHOLD = 2.0
DEFAULT_ATR_MULTIPLIER = 1.5


def daily_bias_at(daily_candles: Sequence[Candle], ts_ms: int,
                  fractal_width: int = 2, sr_lookback: int = 20) -> Bias:
    """1D structural bias using only daily candles CLOSED at/before ts_ms
    (the current 4H bar's close) — no lookahead. Reuses compute_bias
    unchanged; returns NEUTRAL when history is too thin for a swing.

    fractal_width/sr_lookback are compute_bias's own defaults, reused
    as-is (untuned for the daily timeframe — flagged in the build doc).
    """
    closed = [c for c in daily_candles if c.close_time_ms <= ts_ms]
    return compute_bias(closed, fractal_width=fractal_width, sr_lookback=sr_lookback).bias


# ── Part 2: cycle state machine (pure decision helpers) ──
#
# The simulator (backtest.py) owns candle-walking, intrabar stop
# detection, and fee accounting. This module holds only the pure,
# deterministic transition logic + the ATR stop level, so it stays a
# testable leaf with no dependency on the backtest harness.


class CycleState(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Leg:
    direction: str      # "LONG" | "SHORT"
    entry: float
    stop: float         # ATR hard stop, fixed at entry (never widened)
    entry_index: int
    cycle_id: int       # groups legs belonging to one macro cycle


def leg_stop(direction: str, entry: float, atr: float,
             atr_mult: float = DEFAULT_ATR_MULTIPLIER) -> float:
    """ATR hard stop for a leg: below entry for a long, above for a short.
    Every leg carries one (non-negotiable risk requirement) — it caps the
    failure mode where a trend pins Fisher extended and the reset the flip
    waits for never comes."""
    offset = atr_mult * atr
    return entry - offset if direction == "LONG" else entry + offset


def opening_direction(bias: Bias, fisher: float, threshold: float) -> str | None:
    """Pullback entry from FLAT (fresh or re-arm after a stop) while the
    macro bias holds: buy the dip in a bullish bias (Fisher oversold),
    sell the rip in a bearish bias (Fisher overbought). Returns
    'LONG' | 'SHORT' | None. A short is NEVER opened fresh in a bullish
    cycle here — it is only ever reached via the exhaustion flip."""
    if bias == Bias.BULLISH and fisher <= -threshold:
        return "LONG"
    if bias == Bias.BEARISH and fisher >= threshold:
        return "SHORT"
    return None


def is_exhausted(direction: str, fisher: float, threshold: float) -> bool:
    """Exhaustion in the OPEN leg's favor -> flip to bank the move. Long
    exhausts at Fisher >= +threshold, short at Fisher <= -threshold. This
    is the profit-taking cycle event — distinct from a stop-out."""
    if direction == "LONG":
        return fisher >= threshold
    if direction == "SHORT":
        return fisher <= -threshold
    return False


def macro_broken(macro_dir: Bias, bias: Bias) -> bool:
    """The macro thesis no longer holds — the current 1D bias is not the
    direction the cycle opened under (includes flips to NEUTRAL). Triggers
    an immediate force-flatten and ends the cycle."""
    return bias != macro_dir
