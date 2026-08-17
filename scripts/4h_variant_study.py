#!/usr/bin/env python3
"""
4H Variant Study: TSMOM30 + FundMR proxy on 4-hour candles.
Same methodology as the 1D v2 study (causal, costs included).
Compares: 4H TSMOM180 (30-day equivalent), FundMR proxy, and combined.

Usage: python3 scripts/4h_variant_study.py
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
CANDLES_PATH = BASE / "research" / "data" / "BTC_4h_snapshot.json"
OUTPUT = BASE / "research" / "output" / "4h_study_results.json"

# ── parameters ─────────────────────────────────────────────────────────
N_CONFIGS = 10  # for deflated Sharpe

def load_4h_candles():
    with open(CANDLES_PATH) as f:
        data = json.load(f)

    schema = data.get("schema", [])
    close_idx = schema.index("close") if "close" in schema else 5
    candles = data.get("candles", [])
    split = data.get("split_index", int(len(candles) * 0.7))

    closes = np.array([c[close_idx] for c in candles], dtype=float)
    N = len(closes)
    print(f"  Loaded {N} 4H bars (2020→2026)")
    print(f"  IS: {split} bars ({split/(N)*100:.0f}%)")
    print(f"  OOS: {N-split} bars ({(N-split)/N*100:.0f}%)")
    return closes, split, N

# ── position returns (CAUSAL) ──────────────────────────────────────────
def position_returns(prices, pos):
    rets = np.diff(prices) / prices[:-1]
    return rets * pos[:-1]  # pos[i] earns return i→i+1

def trade_count(pos):
    return int(np.sum(np.abs(np.diff(pos)) > 1e-6))

def cost_of_trades(n_bars, n_trades, fee=0.00075):
    """Cost series matching given length: -fee per trade flip."""
    if n_trades == 0:
        return np.zeros(n_bars)
    cost_per_bar = fee * n_trades / n_bars
    return np.full(n_bars, -cost_per_bar)

# ── strategies ─────────────────────────────────────────────────────────
def tsmom(prices, slow=180):
    """TSMOM180 on 4H = 30-day momentum."""
    n = len(prices)
    pos = np.zeros(n)
    for i in range(slow, n):
        mom = prices[i] / prices[i - slow] - 1
        pos[i] = 1.0 if mom > 0 else -1.0
    return pos

def tsmom_conviction(prices, slow=180):
    """TSMOM180 with z-score conviction pyramid."""
    n = len(prices)
    pos = np.zeros(n)
    mom_vals = np.zeros(n)
    for i in range(slow, n):
        mom_vals[i] = prices[i] / prices[i - slow] - 1

    valid = mom_vals[slow:]
    if np.std(valid) > 1e-10:
        mean_mom = np.mean(valid)
        std_mom = np.std(valid)
        z = (mom_vals - mean_mom) / std_mom
        scale = np.minimum(1.5, 0.5 + np.abs(z) * 0.5)
        pos = np.sign(mom_vals) * np.where(scale > 0.25, scale, 0.25)
        pos[:slow] = 0
    else:
        pos[:slow] = 0
        pos[slow:] = np.sign(mom_vals[slow:])
    return pos

def fund_mr_proxy(prices, lookback=360, low_pct=0.3, high_pct=0.7):
    """
    Vol-based funding MR proxy on 4H (360 bars = 60 days).
    """
    n = len(prices)
    pos = np.zeros(n)
    rets = np.diff(prices) / prices[:-1]

    # EMA of absolute returns as ATR% proxy
    ema = np.zeros(n)
    ema[lookback] = np.mean(np.abs(rets[:lookback]))
    decay = 2.0 / (lookback + 1)
    for i in range(lookback + 1, n):
        ema[i] = ema[i-1] * (1 - decay) + np.abs(rets[i-1]) * decay

    for i in range(lookback * 2, n):
        window = ema[i - lookback:i]
        pct = np.sum(window < ema[i]) / len(window)
        if pct < low_pct:
            pos[i] = 1.0   # vol compression → go long
        elif pct > high_pct:
            pos[i] = -0.5  # vol expansion → short vol

    return pos

# ── metrics ────────────────────────────────────────────────────────────
def compute_metrics(rets, label, cost_rets=None):
    if len(rets) < 30:
        return None

    n = len(rets)
    total_rets = np.sum(rets) if cost_rets is None else np.sum(rets + cost_rets)
    ann_ret = total_rets / n * 365 * 6  # 4H bars: 6 per day
    vol = np.std(rets) * np.sqrt(365 * 6) if np.std(rets) > 1e-12 else 0

    if vol < 1e-10:
        return None

    sharpe = ann_ret / vol if vol > 0 else 0
    defl_sharpe = sharpe / np.sqrt(N_CONFIGS) if N_CONFIGS > 1 else sharpe

    # Max drawdown
    if cost_rets is None:
        equity = np.cumprod(1 + rets)
    else:
        equity = np.cumprod(1 + rets + cost_rets)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = np.min(dd)

    net_mult = equity[-1] if len(equity) > 0 else 1.0

    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "defl_sharpe": round(defl_sharpe, 3),
        "ann_ret_pct": round(ann_ret * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "net_mult": round(net_mult, 3),
    }

# ── main ───────────────────────────────────────────────────────────────
def main():
    print("═══ 4H Variant Study ═══")
    print()

    closes, split, N = load_4h_candles()
    print()

    print("── Strategy Computation ──")
    pos_base = tsmom(closes)
    pos_conv = tsmom_conviction(closes)
    pos_fund = fund_mr_proxy(closes)
    pos_combined = np.clip(pos_base + pos_fund, -1.5, 1.5)
    pos_bh = np.ones(N)  # buy and hold

    strategies = {
        "TSMOM180_base": pos_base,
        "TSMOM180_conviction": pos_conv,
        "FundMR_proxy_4H": pos_fund,
        "Combined": pos_combined,
        "BuyHold": pos_bh,
    }

    # IS and OOS returns
    results = {}
    for name, pos in strategies.items():
        rets = position_returns(closes, pos)

        # IS: bars [slow, split) where slow=180 for most strategies
        # Use the same slice for all for fair comparison
        for label, sl, el in [("IS", 0, split - 1), ("OOS", split - 1, len(rets))]:
            r_slice = rets[sl:el]
            if len(r_slice) < 30:
                continue
            n_trades = trade_count(pos[sl if label == "IS" else split:el + 1])
            n_bars = len(r_slice)
            costs = cost_of_trades(n_bars, n_trades)

            m = compute_metrics(r_slice, label, cost_rets=costs)
            if m:
                m["n_trades"] = n_trades
                m_label = f"{name}_{label}"
                results[m_label] = m

    print()
    print("── Results ──")

    # OOS comparison table
    print("\nOOS Comparison (2024-2026):")
    print(f"  {'Strategy':<25} {'Sharpe':>8} {'DeflSh':>8} {'AnnRet%':>8} {'Vol%':>8} {'MaxDD%':>8} {'NetMult':>8} {'Trades':>8}")
    print(f"  {'─'*80}")
    for name in ["TSMOM180_base", "TSMOM180_conviction", "FundMR_proxy_4H", "Combined", "BuyHold"]:
        m = results.get(f"{name}_OOS")
        if m:
            print(f"  {name:<25} {m['sharpe']:>8.3f} {m['defl_sharpe']:>8.3f} {m['ann_ret_pct']:>8.2f} {m['vol_pct']:>8.2f} {m['max_dd_pct']:>8.2f} {m['net_mult']:>8.3f} {m['n_trades']:>8}")

    # IS comparison
    print("\nIS Comparison (2020-2024):")
    print(f"  {'Strategy':<25} {'Sharpe':>8} {'DeflSh':>8} {'AnnRet%':>8} {'NetMult':>8} {'Trades':>8}")
    print(f"  {'─'*68}")
    for name in ["TSMOM180_base", "TSMOM180_conviction", "FundMR_proxy_4H", "Combined", "BuyHold"]:
        m = results.get(f"{name}_IS")
        if m:
            print(f"  {name:<25} {m['sharpe']:>8.3f} {m['defl_sharpe']:>8.3f} {m['ann_ret_pct']:>8.2f} {m['net_mult']:>8.3f} {m['n_trades']:>8}")

    # Save
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n── Saved → {OUTPUT}")

    # Verdict
    combined_oos = results.get("Combined_OOS")
    bh_oos = results.get("BuyHold_OOS")
    if combined_oos and bh_oos:
        beats = combined_oos["sharpe"] > bh_oos["sharpe"]
        print(f"\n── Verdict ──")
        if beats:
            print(f"  ✅ Combined({combined_oos['sharpe']}) beats BuyHold({bh_oos['sharpe']}) on OOS ×{combined_oos['sharpe']/bh_oos['sharpe']:.1f}")
        else:
            print(f"  ❌ Combined({combined_oos['sharpe']}) DOES NOT beat BuyHold({bh_oos['sharpe']}) on OOS")
        print(f"  Trades: {combined_oos['n_trades']} over OOS period")
        print(f"  Deflated Sharpe: {combined_oos['defl_sharpe']}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
