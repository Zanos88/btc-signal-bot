"""Unit tests for strategy.signals.manual_entry_levels.

`manual_entry_levels` powers the control plane's manual Buy/Sell trade
panel (main.py writes its output to market_state; the panel reads it back
to propose strategy-anchored stops/targets). It is a capital-adjacent
path that had no direct coverage — these tests pin its structural
stop/target math and its None-fallback behaviour, and assert it stays
byte-consistent with the automated signal path's structural stop.

Pure/read-only: synthetic BiasResult fixtures, no DB or network.
"""
from __future__ import annotations

from strategy.bias_4h import Bias, BiasResult, SRLevel
from strategy.signals import (
    STRUCTURAL_STOP_BUFFER,
    SignalDirection,
    manual_entry_levels,
    _nearest_resistance,
    _nearest_support,
)


def _bias(fib_levels=None, sr_levels=None):
    return BiasResult(bias=Bias.BULLISH, swing=None,
                      fib_levels=fib_levels or {}, sr_levels=sr_levels or [],
                      reason="synthetic fixture")


def test_both_sides_present():
    b = _bias(fib_levels={"0.618": 96.0, "1.272": 118.0},
              sr_levels=[SRLevel(105.0, "resistance"), SRLevel(95.0, "support")])
    levels = manual_entry_levels(b, price=100.0)

    # long stop = nearest support below price, buffered DOWN by 0.15%
    assert levels["long_stop"] == 95.0 * (1 - STRUCTURAL_STOP_BUFFER)
    # long target = nearest opposing level above price (105 res vs 118 ext)
    assert levels["long_target"] == 105.0
    # short stop = nearest resistance above price, buffered UP by 0.15%
    assert levels["short_stop"] == 105.0 * (1 + STRUCTURAL_STOP_BUFFER)
    # short target = nearest opposing level below price (96 fib vs 95 support)
    assert levels["short_target"] == 96.0


def test_stop_is_none_when_no_structural_level_on_that_side():
    # only a resistance above price -> a long has no support to anchor a
    # stop, a short has no support to target below.
    b = _bias(sr_levels=[SRLevel(105.0, "resistance")])
    levels = manual_entry_levels(b, price=100.0)
    assert levels["long_stop"] is None          # no support below
    assert levels["short_stop"] == 105.0 * (1 + STRUCTURAL_STOP_BUFFER)
    assert levels["long_target"] == 105.0       # resistance above
    assert levels["short_target"] is None       # nothing below


def test_target_is_none_when_no_opposing_level():
    # a lone support below price: a long can anchor a stop but has no
    # level above to target; a short has a target but no resistance stop.
    b = _bias(sr_levels=[SRLevel(95.0, "support")])
    levels = manual_entry_levels(b, price=100.0)
    assert levels["long_stop"] == 95.0 * (1 - STRUCTURAL_STOP_BUFFER)
    assert levels["long_target"] is None
    assert levels["short_stop"] is None
    assert levels["short_target"] == 95.0


def test_empty_bias_yields_all_none():
    levels = manual_entry_levels(_bias(), price=100.0)
    assert levels == {"long_stop": None, "long_target": None,
                      "short_stop": None, "short_target": None}


def test_nearest_support_resistance_pick_the_closest():
    b = _bias(sr_levels=[SRLevel(90.0, "support"), SRLevel(95.0, "support"),
                         SRLevel(105.0, "resistance"), SRLevel(110.0, "resistance")])
    assert _nearest_support(b, 100.0) == 95.0     # closest below, not 90
    assert _nearest_resistance(b, 100.0) == 105.0  # closest above, not 110
    levels = manual_entry_levels(b, price=100.0)
    assert levels["long_stop"] == 95.0 * (1 - STRUCTURAL_STOP_BUFFER)
    assert levels["short_stop"] == 105.0 * (1 + STRUCTURAL_STOP_BUFFER)


def test_manual_long_stop_matches_automated_structural_stop():
    """The manual panel and the automated signal path must anchor a long
    stop to the SAME buffered support — a divergence would mean a manual
    take sizes off a different stop than the alert showed."""
    support = 95.0
    b = _bias(sr_levels=[SRLevel(support, "support"), SRLevel(105.0, "resistance")])
    manual = manual_entry_levels(b, price=100.0)["long_stop"]

    # mirror strategy.signals.evaluate_signal's structural-stop construction
    automated_structural = _nearest_support(b, 100.0) * (1 - STRUCTURAL_STOP_BUFFER)
    assert manual == automated_structural

    # and the short side against a resistance
    manual_short = manual_entry_levels(b, price=100.0)["short_stop"]
    automated_short = _nearest_resistance(b, 100.0) * (1 + STRUCTURAL_STOP_BUFFER)
    assert manual_short == automated_short
