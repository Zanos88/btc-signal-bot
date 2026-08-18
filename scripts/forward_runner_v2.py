#!/usr/bin/env python3
"""
Forward Runner v2 — TSMOM14 for Propr Comp
Deployed as cron every 6h. Generates trade-ready signals with comp-compliant sizing
AND accumulates a real paper equity curve so the champion strategy has forward P&L.

Comp conditions: ~5% daily loss limit, ~10% max total drawdown
Target: 10% account win in 5-6 trading days

Strategy: TSMOM14 (14-day momentum) at up to 2x leverage
Risk scaling: Kelly-optimal fraction with daily-loss ceiling

Equity accumulation: each tick marks the prior position to market against the new
close, applies the taker fee on flips, and appends a mark to
research/output/forward_runner_v2_marks.json. Start equity $100,000.
"""

import json, os, sys, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# ── Config ──
LOOKBACK = 14
LEVERAGE_TARGET = 2.0          # max leverage for comp target
LEVERAGE_MIN = 1.5             # minimum when in position
DAILY_LOSS_LIMIT = 0.05        # 5% max daily drawdown
COMP_TARGET = 0.10             # 10% account win target
REGIME_DAYS = 60               # vol estimation window
TAKER_FEE = 0.00075            # 0.075% per side
START_EQUITY = 100_000.0

# Data paths
DATA_HL = Path("research/data/BTC_1d_snapshot.json")
DATA_BINANCE = Path("/opt/data/candles-binance/BTC_1d_snapshot.json")
STATE_FILE = Path("research/output/forward_runner_v2_state.json")
OUTPUT_FILE = Path("research/output/forward_runner_v2_output.json")
MARKS_FILE = Path("research/output/forward_runner_v2_marks.json")

# ── Helpers ──
def load_candles(path: Path) -> np.ndarray:
    """Load close prices from candle JSON."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    candles = data["candles"]
    return np.array([c[5] for c in candles])  # schema: [... close index 5]

def ts_momentum(prices: np.ndarray, lookback: int) -> float:
    """Return position signal: 1 (long), -1 (short), 0 (flat)."""
    if len(prices) < lookback:
        return 0
    ret = (prices[-1] - prices[-lookback]) / prices[-lookback]
    return 1 if ret > 0 else -1 if ret < 0 else 0

def estimate_vol(prices: np.ndarray, window: int = REGIME_DAYS) -> float:
    """Estimate daily vol from recent returns."""
    if len(prices) < window:
        window = len(prices)
    rets = np.diff(prices[-(window + 1):]) / prices[-(window + 1):-1]
    return np.std(rets, ddof=1) if len(rets) > 1 else 0.03

def comp_sizing(vol: float, signal: int, entry_price: float,
                current_price: float, daily_pnl: float, account_equity: float = 100000.0,
                day_trades: int = 0) -> dict:
    """
    Position sizing for Propr Comp constraints.
    Returns {action, size, limit_price, stop_price, rationale}.
    """
    if signal == 0:
        return {"action": "CLOSE", "size": 0, "rationale": "No signal"}

    # Base position: Kelly-optimal fraction (simplified)
    # For TSMOM14 with Sharpe ~0.93, optimal Kelly = Sharpe / vol
    # But capped by daily loss limit
    kelly_fraction = 0.93 / (vol * np.sqrt(365)) if vol > 0 else 0
    kelly_fraction = min(kelly_fraction, 1.0)  # cap at 1.0

    # Comp constraint: daily loss ≤ 5%
    max_pos_by_daily = (DAILY_LOSS_LIMIT * account_equity) / (vol * account_equity * 3)  # 3σ protection
    max_pos_by_daily = min(max_pos_by_daily, 0.95)  # leave room

    # Target leverage for comp
    target_lev = min(LEVERAGE_TARGET, max_pos_by_daily)

    # Scale: target_lev * kelly_fraction
    size = target_lev * kelly_fraction

    # Don't exceed target leverage
    size = min(size, LEVERAGE_TARGET)

    # If day already has significant PnL, reduce
    if abs(daily_pnl) > 0.02 * account_equity and size > 0:
        size *= 0.5

    direction = "LONG" if signal > 0 else "SHORT"
    notional = size * account_equity
    pos_qty = notional / current_price if current_price > 0 else 0

    return {
        "action": f"ENTER_{direction}",
        "size": round(size, 2),
        "leverage": round(target_lev, 2),
        "notional_usd": round(notional, 2),
        "qty_btc": round(pos_qty, 6),
        "entry_price": current_price,
        "stop_price": round(current_price * (0.97 if signal > 0 else 1.03), 2),
        "daily_loss_room": round(DAILY_LOSS_LIMIT * account_equity - abs(daily_pnl), 2),
        "rationale": f"TSMOM14 signal={direction}, kelly={kelly_fraction:.2f}, lev={target_lev:.1f}x"
    }


def load_json(path: Path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def main():
    print(f"\n═══ Forward Runner v2 — TSMOM14 Comp ═══")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Leverage target: {LEVERAGE_TARGET}x  |  Daily loss limit: {DAILY_LOSS_LIMIT*100:.0f}%")

    base = Path(__file__).resolve().parent.parent

    # ── Load data ──
    # Prefer HL data, fall back to Binance
    prices = load_candles(DATA_HL)
    source = "Hyperliquid"
    if prices is None:
        prices = load_candles(DATA_BINANCE)
        source = "Binance"
    if prices is None:
        print("ERROR: No data available")
        sys.exit(1)

    print(f"  Data: {source} — {len(prices)} bars")

    # ── Signal ──
    close = prices[-1]
    signal = ts_momentum(prices, LOOKBACK)
    vol = estimate_vol(prices)

    # ── Paper equity: mark prior position to market ──
    now = datetime.now(timezone.utc).isoformat()
    prev_state = load_json(base / STATE_FILE, {})
    paper = prev_state.get("paper", {})
    marks = load_json(base / MARKS_FILE, [])

    equity = float(paper.get("equity", START_EQUITY))
    prev_close = paper.get("prev_close")
    prev_signal = paper.get("prev_signal")
    flips = int(paper.get("flips", 0))
    total_fees = float(paper.get("total_fees", 0.0))
    prev_size = float(paper.get("prev_size", 0.0))
    first_mark = paper.get("first_mark_ts", now)

    # If we have a previous close and were in a position, realize the move
    if prev_close is not None and prev_signal is not None and prev_close > 0:
        ret = (close - prev_close) / prev_close
        if prev_signal != 0 and ret != 0:
            equity *= (1 + prev_signal * ret * prev_size)

    # Flip fee when signal changes (or first entry)
    if prev_signal is None:
        # first run: pay entry fee if taking a position
        if signal != 0:
            fee = TAKER_FEE * abs(signal) * equity
            equity -= fee
            total_fees += fee
    elif signal != prev_signal:
        fee = TAKER_FEE * abs(signal - prev_signal) * equity
        equity -= fee
        total_fees += fee
        flips += 1

    # ── Sizing ──
    sizing = comp_sizing(vol, signal, close, close, 0.0, equity)

    # ── State (persist) ──
    state = {
        "current_signal": signal,
        "last_update": now,
        "last_sizing": sizing,
        "paper": {
            "equity": round(equity, 2),
            "prev_close": close,
            "prev_signal": signal,
            "prev_size": sizing["size"],
            "flips": flips,
            "total_fees": round(total_fees, 2),
            "first_mark_ts": first_mark,
        },
    }

    # ── Marks (append) ──
    marks.append({
        "ts": now,
        "close": close,
        "signal": signal,
        "size": sizing["size"],
        "equity": round(equity, 2),
        "total_ret_pct": round((equity / START_EQUITY - 1) * 100, 3),
        "flips": flips,
    })
    # keep the curve bounded
    if len(marks) > 2000:
        marks = marks[-2000:]

    # ── Output ──
    output = {
        "timestamp_utc": now,
        "btc_price": close,
        "volatility_pct": round(vol * 100, 2),
        "signal": "BULL" if signal > 0 else "BEAR" if signal < 0 else "FLAT",
        "lookback": LOOKBACK,
        "leverage": sizing["leverage"],
        "comp_target_pct": COMP_TARGET * 100,
        "daily_loss_limit_pct": DAILY_LOSS_LIMIT * 100,
        "position": {
            "action": sizing["action"],
            "size": sizing["size"],
            "leverage": sizing["leverage"],
            "notional_usd": sizing["notional_usd"],
            "qty_btc": sizing["qty_btc"],
            "entry_price": sizing["entry_price"],
            "stop_price": sizing["stop_price"],
            "daily_loss_room": sizing["daily_loss_room"],
            "rationale": sizing["rationale"]
        },
        "account": {
            "equity": round(equity, 2),
            "start_equity": START_EQUITY,
            "total_return_pct": round((equity / START_EQUITY - 1) * 100, 3),
            "flips": flips,
            "total_fees": round(total_fees, 2),
            "daily_pnl": 0.0,
            "daily_loss_limit": DAILY_LOSS_LIMIT * equity,
            "target_win": COMP_TARGET * equity
        }
    }

    # Save
    with open(base / STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    with open(base / OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    with open(base / MARKS_FILE, "w") as f:
        json.dump(marks, f, indent=2)

    print(f"\n  BTC: ${close:,.0f} | Vol: {vol*100:.1f}% daily | Signal: {output['signal']}")
    print(f"  Size: {sizing['size']:.2f}x ({sizing['leverage']:.2f}x leverage)")
    print(f"  Notional: ${sizing['notional_usd']:,.0f}")
    print(f"  Stop: ${sizing['stop_price']:,.0f}")
    print(f"  Rationale: {sizing['rationale']}")
    print(f"\n  PAPER EQUITY: ${equity:,.2f}  ({output['account']['total_return_pct']:+.2f}% total | {flips} flips | ${total_fees:,.2f} fees)")
    print(f"  Marks: {len(marks)} (→ {MARKS_FILE})")
    print(f"\n  Saved → {STATE_FILE}")
    print(f"         {OUTPUT_FILE}")

    # ── Deliver signal ──
    print(f"\n── DELIVERY ──")
    status = "🟢" if signal > 0 else "🔴" if signal < 0 else "⚪"
    print(f"{status} BTC ${close:,.0f} | TSMOM14 {output['signal']} {sizing['size']:.2f}x ({sizing['leverage']:.2f}x lev) | "
          f"Paper ${equity:,.0f} ({output['account']['total_return_pct']:+.2f}%) | Stop ${sizing['stop_price']:,.0f} | Comp target: {COMP_TARGET*100:.0f}%")

if __name__ == "__main__":
    main()
