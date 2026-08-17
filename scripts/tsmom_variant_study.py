#!/usr/bin/env python3
"""
TSMOM Variant Study — Propr Comp Frequency Analysis

Tests TSMOM7, TSMOM14, TSMOM21, TSMOM30 across:
1. Full-span Sharpe (net of costs, 1x leverage)
2. Win rate in 5-day windows (comp relevance)
3. Expected 5-day return distribution
4. Walk-forward consistency
5. Drawdown under daily loss limits

Comp conditions: ~5% daily loss limit, ~10% max total drawdown
Target: 10% account win in 5-6 trading days
"""

import json, sys, os
import numpy as np
from scipy import stats
from scipy.stats import norm
from pathlib import Path

# ── Config ──
DATA_PATH = "research/data/BTC_1d_snapshot.json"
RESULTS_PATH = "research/output/tsmom_variant_study_results.json"
LOOKBACKS = [7, 14, 21, 30]
REGIME_DAYS = 60  # vol estimation window
LEVERAGE_CAP = 1.0
TAKER_FEE = 0.00075  # 3.15bps * ~2.4 (spread+slip modelled as 0.075%)
N_CONFIGS = 8  # 4 lookbacks × 2 (gross/net)
WALK_FORWARD_FOLDS = 5
COMP_WINDOW = 5  # trading days
COMP_TARGET = 0.10  # 10%
DAILY_LOSS_LIMIT = 0.05  # 5%

def load_data(path: str) -> dict:
    base = Path(__file__).resolve().parent.parent
    full = base / path
    if not full.exists():
        raise FileNotFoundError(f"{full} not found")
    with open(full) as f:
        return json.load(f)

def compute_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(prices) / prices[:-1]

def ts_momentum(lookback: int, rets: np.ndarray) -> np.ndarray:
    """Compound return over lookback window."""
    pos = np.zeros(len(rets))
    for i in range(lookback, len(rets)):
        ret = np.prod(1 + rets[i - lookback:i]) - 1
        pos[i] = 1 if ret > 0 else -1 if ret < 0 else 0
    return pos

def compute_metrics(label: str, rets: np.ndarray, pos: np.ndarray,
                    costs: float = 0.0, ann_factor: int = 365,
                    strategy_prices: int = 0) -> dict:
    """Compute strategy metrics from return series and positions."""
    n = len(rets)
    if n < 30:
        return {"label": label, "error": "insufficient data"}

    # Align: pos[i] is position for bar i, applied to rets[i]
    # Both arrays are same length (both derived from n-1 returns)
    if len(pos) < 30:
        return {"label": label, "error": "insufficient data"}

    strategy_rets = rets[:len(pos)]
    strategy_pos = pos[:len(strategy_rets)]

    # Apply costs
    trade_count = int(np.sum(np.abs(np.diff(strategy_pos)) > 0))
    cost_total = 0.0
    if costs > 0 and trade_count > 0:
        # Costs deducted exactly on flip bars — each flip = one-side taker fee
        flip_bars = np.where(np.abs(np.diff(strategy_pos)) > 0)[0] + 1  # +1 because diff shortens
        for fb in flip_bars:
            if fb < len(strategy_rets):
                strategy_rets[fb] -= costs  # one-side taker fee
        cost_total = costs * trade_count

    # Calculate returns with position
    # position alignment: pos[i] at bar i determines return for bar i
    if len(strategy_pos) > len(strategy_rets):
        strategy_pos = strategy_pos[:len(strategy_rets)]

    bar_rets = strategy_pos * strategy_rets

    # Vol (annualized)
    vol_daily = np.std(bar_rets, ddof=1) if np.std(bar_rets, ddof=1) > 1e-10 else 1e-10
    vol_ann = vol_daily * np.sqrt(ann_factor)

    # Sharpe
    excess = bar_rets - 0.0  # risk-free ~0
    sr_daily = np.mean(excess) / vol_daily if vol_daily > 0 else 0
    sr_ann = sr_daily * np.sqrt(ann_factor)

    # Annualized return
    total_ret = np.prod(1 + bar_rets) - 1
    n_years = len(bar_rets) / ann_factor
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Max drawdown
    cum = np.cumprod(1 + bar_rets)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = np.min(dd)

    # Net multiple
    net_mult = np.prod(1 + bar_rets)

    # DSR (simple)
    dsr_prob = 0.0
    if sr_daily > 0 and vol_daily > 0:
        t_stat = sr_daily * np.sqrt(len(bar_rets))
        dsr_prob = norm.cdf(t_stat / np.sqrt(1 + 2 * ann_factor / len(bar_rets)))

    # Comp metrics
    # Rolling 5-day returns
    comp_returns = []
    for i in range(0, len(bar_rets) - COMP_WINDOW + 1, 1):
        window_ret = np.prod(1 + bar_rets[i:i + COMP_WINDOW]) - 1
        comp_returns.append(window_ret)
    comp_returns = np.array(comp_returns)

    comp_win_rate = np.mean(comp_returns >= COMP_TARGET) if len(comp_returns) > 0 else 0
    comp_mean = np.mean(comp_returns) if len(comp_returns) > 0 else 0
    comp_std = np.std(comp_returns, ddof=1) if len(comp_returns) > 1 else 0
    comp_best = np.max(comp_returns) if len(comp_returns) > 0 else 0
    comp_sharpe_5d = (comp_mean - 0) / comp_std if comp_std > 1e-10 else 0

    # Daily drawdown exceedance
    daily_rets = bar_rets  # already daily
    dd_exceedance = np.mean(daily_rets < -DAILY_LOSS_LIMIT)

    return {
        "label": label,
        "sharpe": round(sr_ann, 3),
        "dsr_prob": round(dsr_prob, 4),
        "ann_ret_pct": round(ann_ret * 100, 2),
        "vol_pct": round(vol_ann * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "net_mult": round(net_mult, 3),
        "n_trades": trade_count,
        "trade_freq_pct": round(trade_count / len(bar_rets) * 100, 2) if len(bar_rets) > 0 else 0,
        "comp_5d_win_rate_pct": round(comp_win_rate * 100, 2),
        "comp_5d_mean_pct": round(comp_mean * 100, 2),
        "comp_5d_std_pct": round(comp_std * 100, 2),
        "comp_5d_best_pct": round(comp_best * 100, 2),
        "comp_5d_sharpe": round(comp_sharpe_5d, 3),
        "daily_loss_exceedance_pct": round(dd_exceedance * 100, 4)
    }

def walk_forward_cv(lookback: int, rets: np.ndarray, costs: float = 0.0):
    """Walk-forward CV with consistent fold sizes."""
    n = len(rets)
    fold_size = n // WALK_FORWARD_FOLDS
    results = []

    for fold in range(WALK_FORWARD_FOLDS):
        val_start = fold * fold_size
        val_end = n if fold == WALK_FORWARD_FOLDS - 1 else (fold + 1) * fold_size

        # Train on data before val_start
        train_rets = rets[:val_start]
        if len(train_rets) < lookback + 20:
            continue

        train_pos = ts_momentum(lookback, train_rets)
        val_pos = ts_momentum(lookback, rets[:val_end])

        # Walk-forward: use train_pos is irrelevant for val; we just use val_pos aligned
        val_rets = rets[val_start:val_end]
        val_pos_aligned = val_pos[val_start:val_end]

        if len(val_pos_aligned) != len(val_rets):
            min_len = min(len(val_pos_aligned), len(val_rets))
            val_pos_aligned = val_pos_aligned[:min_len]
            val_rets = val_rets[:min_len]

        if len(val_rets) < 10:
            continue

        # Costs deducted on flip bars
        trade_count = int(np.sum(np.abs(np.diff(val_pos_aligned)) > 0))
        cost_total = 0.0
        if costs > 0 and trade_count > 0:
            flip_bars = np.where(np.abs(np.diff(val_pos_aligned)) > 0)[0] + 1
            for fb in flip_bars:
                if fb < len(val_rets):
                    val_rets[fb] -= costs
            cost_total = costs * trade_count

        bar_rets = val_pos_aligned * val_rets
        vol = np.std(bar_rets, ddof=1) if np.std(bar_rets, ddof=1) > 1e-10 else 1e-10
        sr = (np.mean(bar_rets) / vol) * np.sqrt(365) if vol > 0 else 0

        results.append({
            "fold": fold + 1,
            "n_bars": len(val_rets),
            "sharpe": round(sr, 3),
            "n_trades": trade_count
        })

    return results


def main():
    print("\n═══ TSMOM Variant Study — Propr Comp Analysis ═══")
    print("Target: 10% account win in 5-day windows")
    print("Constraint: ≤5% daily loss limit\n")

    # Load data
    data = load_data(DATA_PATH)
    candles = data["candles"]
    # schema: ['open_time_ms', 'close_time_ms', 'open', 'high', 'low', 'close', 'volume']
    prices = np.array([c[5] for c in candles])  # close is index 5
    rets = compute_returns(prices)
    print(f"Loaded {len(prices)} bars → {len(rets)} return bars")

    all_results = {}
    config_count = 0

    # ── Full-span comparison ──
    print("\n── Full-Span (1x leverage, net of costs) ──")
    header = f"{'Variant':<18} {'Sharpe':>7} {'DSR':>7} {'AnnRet%':>8} {'Vol%':>7} {'MaxDD%':>8} {'NetMult':>8} {'Trades':>7} {'5dWin%':>7} {'5dMean%':>8} {'5dShrp':>7}"
    print(header)
    print("─" * len(header))

    for lb in LOOKBACKS:
        pos = ts_momentum(lb, rets)

        # Gross
        label_g = f"TSMOM{lb}"
        m_g = compute_metrics(f"TSMOM{lb}_gross", rets.copy(), pos.copy(), costs=0.0)
        config_count += 1

        # Net
        m_n = compute_metrics(f"TSMOM{lb}_net", rets.copy(), pos.copy(), costs=TAKER_FEE)
        config_count += 1

        key_g = f"TSMOM{lb}_gross"
        key_n = f"TSMOM{lb}_net"
        all_results[key_g] = m_g if "error" not in m_g else {}
        all_results[key_n] = m_n if "error" not in m_n else {}

        row = f"TSMOM{lb:<12} {m_n['sharpe']:>7.3f} {m_n['dsr_prob']:>7.4f} {m_n['ann_ret_pct']:>8.2f} {m_n['vol_pct']:>7.2f} {m_n['max_dd_pct']:>8.2f} {m_n['net_mult']:>8.3f} {m_n['n_trades']:>7} {m_n['comp_5d_win_rate_pct']:>7.2f} {m_n['comp_5d_mean_pct']:>8.2f} {m_n['comp_5d_sharpe']:>7.3f}"
        print(row)

    # BuyHold
    bh_pos = np.ones(len(rets))
    bh = compute_metrics("BuyHold", rets.copy(), bh_pos, costs=0.0)
    all_results["BuyHold"] = bh if "error" not in bh else {}
    row = f"{'BuyHold':<18} {bh['sharpe']:>7.3f} {bh['dsr_prob']:>7.4f} {bh['ann_ret_pct']:>8.2f} {bh['vol_pct']:>7.2f} {bh['max_dd_pct']:>8.2f} {bh['net_mult']:>8.3f} {'1':>7} {bh['comp_5d_win_rate_pct']:>7.2f} {bh['comp_5d_mean_pct']:>8.2f} {bh['comp_5d_sharpe']:>7.3f}"
    print(row)

    # ── Walk-forward CV ──
    print("\n── Walk-Forward CV (5 folds) ──")
    for lb in LOOKBACKS:
        cv = walk_forward_cv(lb, rets, costs=TAKER_FEE)
        if cv:
            sharpes = [f["sharpe"] for f in cv]
            trades = [f["n_trades"] for f in cv]
            mean_sr = np.mean(sharpes)
            std_sr = np.std(sharpes, ddof=1) if len(sharpes) > 1 else 0
            mean_tr = np.mean(trades)
            print(f"  TSMOM{lb}:   Sharpe {mean_sr:.3f} ± {std_sr:.3f}   Avg trades/fold: {mean_tr:.0f}")
            all_results[f"TSMOM{lb}_wfcv"] = {
                "fold_sharpes": sharpes,
                "mean_sharpe": round(mean_sr, 3),
                "std_sharpe": round(std_sr, 3),
                "mean_trades_per_fold": round(mean_tr, 0)
            }

    # ── Deflated Sharpe ──
    print("\n── Honest Deflated Sharpe ──")
    all_sharpes = []
    for k, v in all_results.items():
        if "sharpe" in v and "gross" in k:
            all_sharpes.append(abs(v["sharpe"]))
    n_configs = len(all_sharpes)
    best_sr = max(all_sharpes) if all_sharpes else 0
    se_sr = 1 / np.sqrt(2150 - 1)  # approximate SE
    if n_configs > 1 and se_sr > 0:
        e_max = norm.ppf(1 - 0.5 * (1 / n_configs)) * se_sr
        # Find the net Sharpe of the best variant
        best_key = max(((k, v["sharpe"]) for k, v in all_results.items() if "sharpe" in v and "net" in k and v["sharpe"] > 0), key=lambda x: x[1], default=(None, 0))
        best_net_sr = best_key[1]
        dsr = norm.cdf((best_net_sr / np.sqrt(365) - e_max) / se_sr) if best_net_sr > 0 else 0
    else:
        dsr = 0

    print(f"  Configs tried: {n_configs}")
    print(f"  Best net Sharpe: {best_net_sr:.3f}")
    print(f"  DSR across {n_configs} trials: {dsr:.4f}")

    best_variant_name = best_key[0] if best_key[0] else "none"

    # ── Comp Readiness ──
    print("\n── Propr Comp Readiness ──")
    print(f"  Window: {COMP_WINDOW} trading days")
    print(f"  Target: {COMP_TARGET*100:.0f}% account win")
    print(f"  Daily loss limit: {DAILY_LOSS_LIMIT*100:.0f}%")

    best_variant = None
    for lb in LOOKBACKS:
        key = f"TSMOM{lb}_net"
        v = all_results.get(key, {})
        if not v or "error" in v:
            continue
        win_rate = v.get("comp_5d_win_rate_pct", 0)
        mean_5d = v.get("comp_5d_mean_pct", 0)
        dd_exc = v.get("daily_loss_exceedance_pct", 100)
        sharpe = v.get("sharpe", 0)
        trades = v.get("n_trades", 0)

        # Composite score: weight win rate × sharpe × (1 - dd_exceedance)
        comp_score = win_rate * max(0, sharpe) * max(0.01, 1 - dd_exc / 100)

        status = "✅" if win_rate > 5 and dd_exc < 0.5 else "⚠️" if win_rate > 2 else "❌"
        print(f"  {status} TSMOM{lb:>2}:  5dWin={win_rate:.1f}%  Mean5d={mean_5d:.2f}%  DDexc={dd_exc:.3f}%  Score={comp_score:.2f}")

        if best_variant is None or comp_score > best_variant["score"]:
            best_variant = {
                "variant": f"TSMOM{lb}",
                "score": comp_score,
                "win_rate_pct": win_rate,
                "mean_5d_pct": mean_5d,
                "dd_exceedance_pct": dd_exc,
                "sharpe": sharpe,
                "trades": trades
            }

    if best_variant:
        print(f"\n  🏆 Best comp variant: {best_variant['variant']}")
        print(f"     5d win rate: {best_variant['win_rate_pct']:.1f}%")
        print(f"     Mean 5d return: {best_variant['mean_5d_pct']:.2f}%")
        print(f"     Daily loss exceedance: {best_variant['dd_exceedance_pct']:.3f}%")
        print(f"     Sharpe: {best_variant['sharpe']:.3f}")

    # ── Verdict ──
    print("\n── Verdict ──")
    best_tsmom = None
    for lb in LOOKBACKS:
        key = f"TSMOM{lb}_net"
        v = all_results.get(key, {})
        if v and "error" not in v and v.get("sharpe", 0) > 0:
            if best_tsmom is None or v["sharpe"] > best_tsmom["sharpe"]:
                best_tsmom = {"variant": f"TSMOM{lb}", **v}

    if best_tsmom:
        v = best_tsmom
        print(f"  Best overall: {best_tsmom['variant']} | Sharpe {v['sharpe']} | DSR {v['dsr_prob']}")
        print(f"  Net mult: {v['net_mult']}x | Max DD: {v['max_dd_pct']}% | Trades: {v['n_trades']}")

        # Compare to B&H
        bh_sharpe = bh.get("sharpe", 0)
        bh_dd = bh.get("max_dd_pct", 0)
        print(f"  vs BuyHold: Sharpe {v['sharpe']} vs {bh_sharpe} (Δ={v['sharpe'] - bh_sharpe:+.3f})")
        print(f"  vs BuyHold: MaxDD {v['max_dd_pct']}% vs {bh_dd}% (Δ={v['max_dd_pct'] - bh_dd:+.1f}pp)")

    # Comp feasibility
    print(f"\n  Propr comp feasibility:")
    if best_variant:
        # At 1x leverage
        win_once = best_variant["win_rate_pct"] / 100
        mean_5d = best_variant["mean_5d_pct"] / 100
        dd_exc = best_variant["dd_exceedance_pct"] / 100

        print(f"    1x leverage: {win_once*100:.1f}% historical chance of 10% in 5 days")
        print(f"    Mean 5-day return: {mean_5d*100:.2f}%")

        if win_once > 0.05:
            print(f"    ✅ Viable at 1x: ~1-in-{1/win_once:.0f} chance per 5-day window")
            # Optimal leverage
            max_leverage = DAILY_LOSS_LIMIT / (np.std(bh.get("ann_ret_pct", 50) / 100 / np.sqrt(365)) if "ann_ret_pct" in bh else 0.03)
            print(f"    Max safe leverage: ~{max_leverage:.1f}x (daily loss limit constraint)")
        else:
            need_leverage = COMP_TARGET / max(mean_5d, 0.001)
            print(f"    ❌ 1x insufficient. Need ~{need_leverage:.1f}x leverage for expected 10%")

    # Penalty analysis for comp
    print(f"\n  Fill penalty sensitivity:")
    for penalty_bps in [0, 3.15, 6.3, 9.45, 12.6, 18.0]:
        cost = penalty_bps / 10000
        for lb in LOOKBACKS:
            pos = ts_momentum(lb, rets)
            m = compute_metrics(f"TSMOM{lb}_penalty", rets.copy(), pos.copy(), costs=cost)
            if "error" not in m:
                all_results[f"TSMOM{lb}_penalty_{int(penalty_bps*100)}bps"] = {"cost_bps": penalty_bps, "sharpe": m["sharpe"], "n_trades": m["n_trades"]}
                break
        if penalty_bps == 18.0:
            print(f"    18bps m-a-i-k penalty: Sharpe {m['sharpe']:.3f} ({m['n_trades']} trades)")
        elif penalty_bps == 0:
            print(f"    0bps (gross):     Sharpe {m['sharpe']:.3f} ({m['n_trades']} trades)")
        elif penalty_bps == 3.15:
            print(f"    3.15bps (taker):  Sharpe {m['sharpe']:.3f} ({m['n_trades']} trades)")
        elif penalty_bps == 9.45:
            print(f"    9.45bps (mid):    Sharpe {m['sharpe']:.3f} ({m['n_trades']} trades)")

    # ── Save ──
    output = {
        "metadata": {
            "n_bars": len(prices),
            "n_rets": len(rets),
            "comp_window": COMP_WINDOW,
            "comp_target_pct": COMP_TARGET * 100,
            "daily_loss_limit_pct": DAILY_LOSS_LIMIT * 100,
            "taker_fee_pct": TAKER_FEE * 100,
            "n_configs_tested": config_count,
            "deflated_dsr": round(dsr, 4),
            "best_variant": best_variant["variant"] if best_variant else "none",
            "best_variant_score": round(best_variant["score"], 2) if best_variant else 0
        },
        "results": all_results
    }

    base = Path(__file__).resolve().parent.parent
    output_path = base / RESULTS_PATH
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {output_path}")
    print()

if __name__ == "__main__":
    main()
