"""Confluence logic: combines 4H bias + 1H trigger into a gated Signal.

Zero imports from alerts/ or execution/ (build spec section 7) — a Signal
is a plain data object; delivery and (future) execution are consumers of
it, not producers. Implements docs/STRATEGY_PSEUDOCODE.md's entry/exit
decision tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from data.feed import Candle
from strategy.atr import wilder_atr
from strategy.bias_4h import Bias, BiasResult, compute_bias
from strategy.ichimoku import evaluate_ichimoku
from strategy.rsi import Vote, evaluate_rsi
from strategy.trigger_1h import TriggerResult, evaluate_trigger

MIN_REWARD_RISK = 2.0
STRUCTURAL_STOP_BUFFER = 0.0015  # 0.15% beyond the S/R/swing level
DEFAULT_ATR_MULTIPLIER = 1.5     # hybrid stop's ATR floor factor (sweep-tuned, not final)
STOP_MODELS = ("structural", "hybrid")
# |4H Fisher| at/beyond this marks the move as exhausted — a hard rule
# from the user's live discretionary trading (confirmed 2026-07-08);
# the sweep brackets it with 1.5/2.5 as sensitivity checks.
FISHER4H_EXHAUSTION_THRESHOLD = 2.0

# V2.3 target models. The V2.2 sweep showed nearest-structure targets vs
# the R:R>=2 gate are the system's binding constraint (0 trades in 2.3yr
# on 1d/4h) — these extend targets the same way hybrid stops widened
# stops. Live default stays nearest_structure until the sweep is
# reviewed (same rollout discipline as stop models).
TARGET_MODELS = ("nearest_structure", "fib_extension_preferred", "blue_sky_atr")
FIB_EXTENSION_KEYS = ("1.272", "1.618")   # keys in BiasResult.fib_levels
DEFAULT_BLUE_SKY_ATR_MULTIPLIER = 3.0     # unswept first guess (V2.3 open item 1)

# All confluence indicators, independently toggleable. Defaults preserve
# the original 3-indicator behavior exactly (RSI/Ichimoku off until the
# user enables them via /settings -> Indicators).
INDICATOR_NAMES = ("bias_sr", "fisher", "obv", "rsi", "ichimoku")
DEFAULT_INDICATOR_CONFIG = {
    "bias_sr": True, "fisher": True, "obv": True, "rsi": False, "ichimoku": False,
}


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class Signal:
    direction: SignalDirection
    entry: float
    stop: float
    target: float
    reward_risk: float
    timestamp: datetime
    bias_reason: str
    trigger_reason: str


@dataclass(frozen=True)
class SuppressedSignal:
    direction: SignalDirection
    reward_risk: float
    reason: str
    kind: str = "rr"  # "rr" | "fisher4h_exhaustion" | "exhaustion_standdown" — lets callers count suppression classes separately


def resolve_stop(
    direction: SignalDirection,
    entry_price: float,
    structural_stop: float,
    trigger_candles: Sequence[Candle],
    stop_model: str = "structural",
    atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
) -> float:
    """Final stop price under the configured stop model.

    "structural": the buffered S/R level unchanged (live default).
    "hybrid": the WIDER of structural vs an ATR floor — min() for longs,
    max() for shorts — so the stop clears trigger-TF noise (V2.2: the
    2026-07-08 backtest showed structural stops of 0.15-0.35% of price
    resolving inside a single trigger bar). Widens only, never tightens;
    falls back to structural when ATR history is insufficient (ATR==0).

    Sizing MUST consume the value returned here via Signal.stop — never
    the pre-hybrid structural level — or realized R drifts from nominal
    R (the R-Drift trap from the V2.2 research breakdown).
    """
    if stop_model not in STOP_MODELS:
        raise ValueError(f"unknown stop_model {stop_model!r} — allowed: {STOP_MODELS}")
    if stop_model == "structural":
        return structural_stop
    atr = wilder_atr(trigger_candles)[-1] if trigger_candles else 0.0
    if atr <= 0.0:
        return structural_stop
    if direction == SignalDirection.LONG:
        return min(structural_stop, entry_price - atr_multiplier * atr)
    return max(structural_stop, entry_price + atr_multiplier * atr)


def _next_opposing_level_above(bias_result: BiasResult, price: float) -> float | None:
    candidates = [lv for lv in bias_result.fib_levels.values() if lv > price]
    candidates += [lv.price for lv in bias_result.sr_levels if lv.kind == "resistance" and lv.price > price]
    return min(candidates) if candidates else None


def _next_opposing_level_below(bias_result: BiasResult, price: float) -> float | None:
    candidates = [lv for lv in bias_result.fib_levels.values() if lv < price]
    candidates += [lv.price for lv in bias_result.sr_levels if lv.kind == "support" and lv.price < price]
    return max(candidates) if candidates else None


def resolve_target(
    direction: SignalDirection,
    entry_price: float,
    bias_result: BiasResult,
    trigger_candles: Sequence[Candle],
    target_model: str = "nearest_structure",
    blue_sky_atr_multiplier: float = DEFAULT_BLUE_SKY_ATR_MULTIPLIER,
) -> float | None:
    """Final target price under the configured target model; None means
    "no defensible target" and the caller drops the signal (unchanged
    from pre-V2.3 behavior).

    "nearest_structure": today's behavior — the nearest opposing level
    (fibs + S/R pooled), unchanged.
    "fib_extension_preferred": if a Fib extension (1.272/1.618) lies
    beyond the nearest opposing level, prefer it — the farther of the
    two candidates — but never through an intervening structural S/R
    level between them (cap at that level instead).
    "blue_sky_atr": fib_extension_preferred, plus a volatility
    projection (entry +/- multiplier*ATR on the trigger TF) when NO
    opposing level exists at all — price beyond all known reference
    levels. Cumulative by design so Grid C isolates each increment:
    model2-vs-1 = extension effect, model3-vs-2 = blue-sky effect.
    Returns None (no bare guess) if ATR history is insufficient.
    """
    if target_model not in TARGET_MODELS:
        raise ValueError(f"unknown target_model {target_model!r} — allowed: {TARGET_MODELS}")

    is_long = direction == SignalDirection.LONG
    nearest = (_next_opposing_level_above if is_long else _next_opposing_level_below)(
        bias_result, entry_price)

    if target_model == "nearest_structure":
        return nearest

    if nearest is None:
        # no opposing level anywhere (extensions are part of the nearest
        # pool, so none exist either) — blue sky, or no target at all
        if target_model != "blue_sky_atr":
            return None
        atr = wilder_atr(trigger_candles)[-1] if trigger_candles else 0.0
        if atr <= 0.0:
            return None
        offset = blue_sky_atr_multiplier * atr
        return entry_price + offset if is_long else entry_price - offset

    # extension preference: nearest fib extension strictly beyond the
    # nearest opposing level, capped at any structural S/R between them
    extensions = [bias_result.fib_levels[k] for k in FIB_EXTENSION_KEYS
                  if k in bias_result.fib_levels]
    if is_long:
        beyond = [e for e in extensions if e > nearest]
        if not beyond:
            return nearest
        ext = min(beyond)
        bounds = [lv.price for lv in bias_result.sr_levels
                  if lv.kind == "resistance" and nearest < lv.price < ext]
        return min(bounds) if bounds else ext
    beyond = [e for e in extensions if e < nearest]
    if not beyond:
        return nearest
    ext = max(beyond)
    bounds = [lv.price for lv in bias_result.sr_levels
              if lv.kind == "support" and ext < lv.price < nearest]
    return max(bounds) if bounds else ext


def _nearest_support(bias_result: BiasResult, price: float) -> float | None:
    supports = [lv.price for lv in bias_result.sr_levels if lv.kind == "support"]
    return max((p for p in supports if p < price), default=None)


def _nearest_resistance(bias_result: BiasResult, price: float) -> float | None:
    resistances = [lv.price for lv in bias_result.sr_levels if lv.kind == "resistance"]
    return min((p for p in resistances if p > price), default=None)


def manual_entry_levels(bias_result: BiasResult, price: float) -> dict:
    """Structural stop/target proposals for MANUAL entries in both
    directions, from the current bias state — consumed by the control
    plane's trade panel (strategy-anchored Buy/Sell). Values are None when
    no structural level exists on the required side; manual entries then
    fall back to the custom-stop path. Same levels/buffer as the
    automated path — no separate logic."""
    support = _nearest_support(bias_result, price)
    resistance = _nearest_resistance(bias_result, price)
    return {
        "long_stop": support * (1 - STRUCTURAL_STOP_BUFFER) if support is not None else None,
        "long_target": _next_opposing_level_above(bias_result, price),
        "short_stop": resistance * (1 + STRUCTURAL_STOP_BUFFER) if resistance is not None else None,
        "short_target": _next_opposing_level_below(bias_result, price),
    }


def evaluate_confluence(
    candles_bias: Sequence[Candle],
    candles_trigger: Sequence[Candle],
    config: dict | None = None,
    ichimoku_variant: str = "standard",
) -> tuple[SignalDirection | None, dict, BiasResult]:
    """Dynamic confluence: entry direction requires ALL ENABLED indicators
    to agree. Disabling an indicator removes it from the requirement.

    One gate function, no per-indicator duplication: every indicator
    contributes a Vote; alignment is computed over the enabled subset.
    Fisher's vote exists only on its cross bar, which makes the default
    config (bias+fisher+obv) behave exactly like the original 3-indicator
    logic. Returns (direction|None, per-indicator readings, bias_result).
    bias_result is always computed — structural stops/targets and
    market_state need it even when its vote is disabled.
    """
    cfg = {**DEFAULT_INDICATOR_CONFIG, **(config or {})}

    bias_result = compute_bias(candles_bias)
    trigger_result: TriggerResult = evaluate_trigger(candles_trigger)
    rsi_reading = evaluate_rsi(candles_trigger)
    ichi = evaluate_ichimoku(candles_bias, variant=ichimoku_variant)

    bias_vote = ("LONG" if bias_result.bias == Bias.BULLISH
                 else "SHORT" if bias_result.bias == Bias.BEARISH else "NONE")
    fisher_vote = ("LONG" if trigger_result.fisher_cross == "bullish"
                   else "SHORT" if trigger_result.fisher_cross == "bearish" else "NONE")
    obv_vote = ("LONG" if trigger_result.obv_confirmation == "rising"
                else "SHORT" if trigger_result.obv_confirmation == "falling" else "NONE")

    readings = {
        "bias_sr": {"enabled": cfg["bias_sr"], "vote": bias_vote,
                    "bias": bias_result.bias.value, "reason": bias_result.reason},
        "fisher": {"enabled": cfg["fisher"], "vote": fisher_vote,
                   "cross": trigger_result.fisher_cross, "value": trigger_result.fisher_value},
        "obv": {"enabled": cfg["obv"], "vote": obv_vote,
                "state": trigger_result.obv_confirmation, "value": trigger_result.obv_value},
        "rsi": {"enabled": cfg["rsi"], "vote": rsi_reading.vote.value, "value": rsi_reading.value},
        "ichimoku": {"enabled": cfg["ichimoku"], "vote": ichi.vote.value,
                     "tenkan": ichi.tenkan, "kijun": ichi.kijun,
                     "senkou_a": ichi.senkou_a, "senkou_b": ichi.senkou_b,
                     "variant": ichi.variant},
    }

    enabled_votes = [r["vote"] for name, r in readings.items() if cfg[name]]
    if not enabled_votes:
        return None, readings, bias_result  # nothing enabled -> never signal
    if all(v == "LONG" for v in enabled_votes):
        return SignalDirection.LONG, readings, bias_result
    if all(v == "SHORT" for v in enabled_votes):
        return SignalDirection.SHORT, readings, bias_result
    return None, readings, bias_result


def standdown_suppresses(direction, funding_pctile: float, threshold: float,
                         oi_z: float | None = None, oi_z_min: float | None = None) -> bool:
    """Trend-exhaustion stand-down (OI×liquidation research §2.3c, BACKTEST-
    ONLY): a trend running on funding-extreme, OI-maxed leverage is crowded,
    not conviction. Suppress entries only in the CROWDED direction — longs
    when funding is at the top of its trailing distribution, shorts at the
    bottom. When oi_z_min is set the condition is a conjunction with the OI
    z-score; when unset (OI data unavailable, Phase 0 §0.2) funding alone
    decides. Pure so it can be unit-tested directly."""
    crowded = (funding_pctile >= threshold if direction == SignalDirection.LONG
               else funding_pctile <= 100.0 - threshold)
    if not crowded:
        return False
    if oi_z_min is not None:
        # Conjunction requires BOTH legs CONFIRMED. When OI is unavailable
        # for this bar (e.g. the bar predates OI-history coverage) the
        # leverage leg cannot be confirmed, so we do NOT stand down — a
        # stand-down acts only on evidence, never on a missing input.
        if oi_z is None:
            return False
        return oi_z >= oi_z_min
    return True


def evaluate_signal(
    candles_4h: Sequence[Candle],
    candles_1h: Sequence[Candle],
    now: datetime | None = None,
    config: dict | None = None,
    ichimoku_variant: str = "standard",
    return_readings: bool = False,
    stop_model: str = "structural",
    atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
    target_model: str = "nearest_structure",
    blue_sky_atr_multiplier: float = DEFAULT_BLUE_SKY_ATR_MULTIPLIER,
    fisher4h_entry_filter: bool = False,
    fisher4h_value: float | None = None,
    exhaustion_threshold: float = FISHER4H_EXHAUSTION_THRESHOLD,
    standdown_entry_filter: bool = False,
    funding_pctile_value: float | None = None,
    funding_pctile_threshold: float = 85.0,
    oi_z_value: float | None = None,
    oi_z_min: float | None = None,
):
    """Full confluence + exit + R:R gate.

    Returns Signal | SuppressedSignal | None (default), or a
    (result, readings) tuple when return_readings=True — readings are the
    per-indicator snapshot destined for indicators_snapshot JSONB.

    stop_model/atr_multiplier select the stop construction (see
    resolve_stop); target_model/blue_sky_atr_multiplier select the
    target construction (see resolve_target). Defaults preserve the
    pre-V2.2/V2.3 nearest-structure behavior exactly — the live engine
    passes nothing here until sweep results are reviewed (user
    decisions 2026-07-08).

    fisher4h_entry_filter (BACKTEST-ONLY until sweep results say
    otherwise): suppress a fresh signal when the 4H Fisher is already
    extended past exhaustion_threshold in the SAME direction — don't
    chase a 4H move that looks capped. The caller computes and passes
    the last-closed 4H Fisher value (this function stays pure); the
    check runs AFTER the R:R gate so its suppression count isolates
    "trades that would otherwise have been taken".
    """
    direction, readings, bias_result = evaluate_confluence(
        candles_4h, candles_1h, config=config, ichimoku_variant=ichimoku_variant,
    )

    def _ret(result):
        return (result, readings) if return_readings else result

    if direction is None:
        return _ret(None)

    # Entry decision on close; entry price is that same close — no
    # intra-candle fill assumption (build spec section 11).
    entry_price = candles_1h[-1].close

    if direction == SignalDirection.LONG:
        support = _nearest_support(bias_result, entry_price)
        if support is None:
            return _ret(None)  # no structural level to anchor a stop — never a bare-percentage stop
        structural_stop = support * (1 - STRUCTURAL_STOP_BUFFER)
    else:
        resistance = _nearest_resistance(bias_result, entry_price)
        if resistance is None:
            return _ret(None)
        structural_stop = resistance * (1 + STRUCTURAL_STOP_BUFFER)

    target = resolve_target(direction, entry_price, bias_result, candles_1h,
                            target_model=target_model,
                            blue_sky_atr_multiplier=blue_sky_atr_multiplier)
    if target is None:
        return _ret(None)  # no defensible target under this model

    # R:R, the Signal, and (downstream) sizing all use the FINAL resolved
    # stop — a hybrid widening must flow through everything or realized R
    # silently drifts from nominal R.
    stop = resolve_stop(direction, entry_price, structural_stop, candles_1h,
                        stop_model=stop_model, atr_multiplier=atr_multiplier)

    risk = abs(entry_price - stop)
    reward = abs(target - entry_price)
    if risk == 0:
        return _ret(None)
    rr = reward / risk

    if rr < MIN_REWARD_RISK:
        return _ret(SuppressedSignal(direction, rr, f"R:R {rr:.2f} below minimum {MIN_REWARD_RISK}"))

    if fisher4h_entry_filter:
        if fisher4h_value is None:
            raise ValueError("fisher4h_entry_filter requires fisher4h_value from the caller")
        extended_same_direction = (
            fisher4h_value >= exhaustion_threshold if direction == SignalDirection.LONG
            else fisher4h_value <= -exhaustion_threshold
        )
        if extended_same_direction:
            return _ret(SuppressedSignal(
                direction, rr,
                f"4H Fisher {fisher4h_value:+.2f} already extended beyond "
                f"+/-{exhaustion_threshold} in {direction.value} direction — not chasing",
                kind="fisher4h_exhaustion",
            ))

    if standdown_entry_filter:
        if funding_pctile_value is None:
            raise ValueError("standdown_entry_filter requires funding_pctile_value from the caller")
        if standdown_suppresses(direction, funding_pctile_value, funding_pctile_threshold,
                                oi_z_value, oi_z_min):
            return _ret(SuppressedSignal(
                direction, rr,
                f"exhaustion stand-down: funding pctile {funding_pctile_value:.0f} "
                f"crowded toward {direction.value}"
                + (f", OI z {oi_z_value:+.2f} >= {oi_z_min}" if oi_z_min is not None else "")
                + " — trend living on borrowed leverage, no fresh entry",
                kind="exhaustion_standdown",
            ))

    active = [f"{name}:{r['vote']}" for name, r in readings.items() if r["enabled"]]
    return _ret(Signal(
        direction=direction,
        entry=entry_price,
        stop=stop,
        target=target,
        reward_risk=rr,
        timestamp=now or datetime.now(timezone.utc),
        bias_reason=bias_result.reason,
        trigger_reason="confluence: " + " + ".join(active),
    ))
