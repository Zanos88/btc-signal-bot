"""Unit tests for strategy/bias_4h.py — the 4H structural engine.

Previously the fractal swing detection, Fibonacci construction, S/R
clustering and `compute_bias` decision were exercised only indirectly
(through backtest replays and hand-built `BiasResult` objects). These
tests pin the pure structural logic directly against synthetic candles,
so no network access is involved.

The candle series are hand-shaped so exactly one fractal low and one
fractal high are confirmed (width=2), yielding a single, predictable
swing leg whose direction and levels the assertions can nail down.
"""
from __future__ import annotations

import pytest

from data.feed import Candle
from strategy.bias_4h import (
    Bias,
    SRLevel,
    Swing,
    SwingDirection,
    _closest,
    _is_fractal_high,
    _is_fractal_low,
    compute_bias,
    detect_swings,
    fibonacci_levels,
    horizontal_sr,
)


def _c(high: float, low: float, close: float) -> Candle:
    """Minimal candle; time fields are unused by the structural logic
    (detect_swings orders by list index, not timestamp)."""
    return Candle(open_time_ms=0, close_time_ms=0, open=close,
                  high=high, low=low, close=close, volume=1.0)


def _up_swing_series(last_close: float, last_high: float = 16, last_low: float = 13):
    """Series with a single confirmed UP swing (fractal low @idx2 -> fractal
    high @idx6). The final candle's close/high/low are parameterized so the
    caller can place price above or below the 0.618 retrace."""
    highs = [12, 14, 11, 14, 16, 18, 22, 17, last_high]
    lows = [10, 8, 5, 8, 10, 12, 15, 14, last_low]
    closes = [11, 11, 8, 11, 13, 15, 18, 15, last_close]
    return [_c(h, low, cl) for h, low, cl in zip(highs, lows, closes)]


def _down_swing_series(last_close: float, last_high: float = 13, last_low: float = 8):
    """Series with a single confirmed DOWN swing (fractal high @idx2 ->
    fractal low @idx6)."""
    highs = [16, 18, 22, 18, 16, 14, 11, 12, last_high]
    lows = [13, 14, 11, 8, 6, 5, 2, 6, last_low]
    closes = [15, 16, 18, 14, 12, 10, 6, 9, last_close]
    return [_c(h, low, cl) for h, low, cl in zip(highs, lows, closes)]


# ── fractal detection ──

def test_fractal_high_and_low_detection():
    candles = _up_swing_series(last_close=15)
    # Only idx6 is a confirmed fractal high, only idx2 a confirmed fractal low.
    assert [i for i in range(len(candles)) if _is_fractal_high(candles, i, 2)] == [6]
    assert [i for i in range(len(candles)) if _is_fractal_low(candles, i, 2)] == [2]


def test_fractal_edges_never_confirmed():
    candles = _up_swing_series(last_close=15)
    # Within `width` of either edge there is no full window -> never a fractal.
    assert _is_fractal_high(candles, 0, 2) is False
    assert _is_fractal_high(candles, 1, 2) is False
    assert _is_fractal_low(candles, len(candles) - 1, 2) is False
    assert _is_fractal_low(candles, len(candles) - 2, 2) is False


def test_swings_do_not_repaint_on_the_final_bar():
    # A fresh extreme printed on the last candle cannot be a swing point yet
    # (needs `width` bars to close after it), so detect_swings ignores it.
    candles = _up_swing_series(last_close=30, last_high=99, last_low=13)
    swings = detect_swings(candles, fractal_width=2)
    assert all(s.end_index <= len(candles) - 3 for s in swings)
    assert not any(s.end_price == 99 for s in swings)


# ── swing legs ──

def test_detect_swings_single_up_leg():
    swings = detect_swings(_up_swing_series(last_close=15), fractal_width=2)
    assert len(swings) == 1
    leg = swings[0]
    assert leg.direction == SwingDirection.UP
    assert leg.start_price == 5 and leg.end_price == 22 and leg.end_index == 6


def test_detect_swings_single_down_leg():
    swings = detect_swings(_down_swing_series(last_close=10), fractal_width=2)
    assert len(swings) == 1
    leg = swings[0]
    assert leg.direction == SwingDirection.DOWN
    assert leg.start_price == 22 and leg.end_price == 2 and leg.end_index == 6


# ── fibonacci + S/R helpers ──

def test_fibonacci_levels_retracement_and_extension():
    swing = Swing(start_price=100.0, end_price=200.0, direction=SwingDirection.UP, end_index=5)
    levels = fibonacci_levels(swing)  # span = 100
    assert levels["0.618"] == pytest.approx(200 - 100 * 0.618)   # retrace back down
    assert levels["0.5"] == pytest.approx(150.0)
    assert levels["1.272"] == pytest.approx(200 + 100 * 0.272)   # extension beyond
    assert levels["1.618"] == pytest.approx(200 + 100 * 0.618)


def test_horizontal_sr_maps_direction_and_respects_lookback():
    swings = [
        Swing(1, 10, SwingDirection.UP, 1),     # high -> resistance @10
        Swing(10, 3, SwingDirection.DOWN, 2),   # low  -> support @3
        Swing(3, 12, SwingDirection.UP, 3),     # high -> resistance @12
    ]
    levels = horizontal_sr(swings, lookback=2)   # only the last 2 swings
    assert levels == [SRLevel(3, "support"), SRLevel(12, "resistance")]


def test_closest_filters_by_kind_and_distance():
    levels = [SRLevel(90, "support"), SRLevel(80, "support"), SRLevel(120, "resistance")]
    assert _closest(levels, 100, "support") == SRLevel(90, "support")
    assert _closest(levels, 100, "resistance") == SRLevel(120, "resistance")
    assert _closest([], 100, "support") is None


# ── compute_bias decision ──

def test_compute_bias_neutral_without_swing():
    # A strictly increasing series has no interior fractal (the window max
    # is always the right edge, the min the left edge) -> no swing -> NEUTRAL.
    rising = [_c(high=10 + i, low=8 + i, close=9 + i) for i in range(8)]
    result = compute_bias(rising)
    assert result.bias == Bias.NEUTRAL
    assert result.swing is None
    assert "no confirmed swing" in result.reason


def test_compute_bias_bullish_when_price_holds_above_618():
    result = compute_bias(_up_swing_series(last_close=15))
    assert result.bias == Bias.BULLISH
    assert result.swing is not None and result.swing.direction == SwingDirection.UP
    assert result.fib_levels["0.618"] == pytest.approx(22 - 17 * 0.618)


def test_compute_bias_neutral_up_swing_below_618():
    # Same up-swing structure, but the last close sits below the 0.618 retrace.
    result = compute_bias(_up_swing_series(last_close=10, last_high=11, last_low=9))
    assert result.bias == Bias.NEUTRAL
    assert "below 0.618" in result.reason or "lost support" in result.reason


def test_compute_bias_bearish_when_price_holds_below_618():
    result = compute_bias(_down_swing_series(last_close=10))
    assert result.bias == Bias.BEARISH
    assert result.swing is not None and result.swing.direction == SwingDirection.DOWN
