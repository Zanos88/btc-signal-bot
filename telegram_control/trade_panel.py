"""Manual trade panel — Trojan-density layout adapted for BTC-PERP.

Layout (per user spec, 2026-07-07):
    [Buy (risk $500)] [Buy (risk $1000)] [Buy X ✏️]
    [✅ BTC-PERP]                     <- single asset; row kept for future
    [Sell 50%] [Sell 100%]
    [Sell Initials] [Sell X% ✏️]
    <position P&L summary line in the panel text, not a sort toggle>
    [🎯 Custom-Stop Buy]
    [← Back] [↻ Refresh]

Buy presets are RISK BUDGETS in USD (mapped to risk-% for the gate),
never notional — fixed-notional buttons remain rejected per the V2
report. Strategy-anchored default: stop/target from the engine-published
market_state structural levels; the custom-stop submenu is the fallback
when no ≥2:1 structure exists. EVERY buy routes through the risk gate;
sells are risk-reducing (reduceOnly) and gate-exempt by design.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal

from risk.sizing import MAX_RISK_PCT, MIN_RISK_PCT
from strategy.signals import Signal, SignalDirection
from telegram_control.handlers import (
    ControlServices,
    _ack,
    _denied,
    _reply,
    _safe_positions,
    _user_id,
)

logger = logging.getLogger(__name__)

BUY_PRESETS_USD = (500, 1000)
CUSTOM_STOP_DISTANCES_PCT = (0.5, 1.0, 1.5, 2.0)
TARGET_MULTIPLE = 2.1  # custom-stop target = 2.1x stop distance (clears the 2:1 gate with margin)

# In-process pending free-text inputs (ForceReply flows), keyed by user id.
# Single-process control plane — no cross-process state needed.
PENDING_INPUTS: dict[int, str] = {}


# ── markups ──

def trade_panel_markup() -> dict:
    return {"inline_keyboard": [
        [
            {"text": "Buy (risk $500)", "callback_data": "trd_buy_500"},
            {"text": "Buy (risk $1000)", "callback_data": "trd_buy_1000"},
            {"text": "Buy X ✏️", "callback_data": "trd_buy_custom"},
        ],
        [{"text": "✅ BTC-PERP", "callback_data": "trd_asset_BTC"}],
        [
            {"text": "Sell 50%", "callback_data": "close_50_BTC"},
            {"text": "Sell 100%", "callback_data": "close_100_BTC"},
        ],
        [
            {"text": "Sell Initials", "callback_data": "trd_sellinit_BTC"},
            {"text": "Sell X% ✏️", "callback_data": "trd_sell_custom"},
        ],
        [{"text": "\U0001F3AF Custom-Stop Buy", "callback_data": "trd_cstopmenu"}],
        [
            {"text": "← Back", "callback_data": "menu_main"},
            {"text": "↻ Refresh", "callback_data": "trd_refresh"},
        ],
    ]}


def custom_stop_markup() -> dict:
    row = [{"text": f"-{d}% stop", "callback_data": f"trd_cstop_{d}"} for d in CUSTOM_STOP_DISTANCES_PCT]
    return {"inline_keyboard": [row, [{"text": "← Back", "callback_data": "trd_refresh"}]]}


def main_menu_markup() -> dict:
    return {"inline_keyboard": [
        [
            {"text": "\U0001F6D2 Trade", "callback_data": "menu_trade"},
            {"text": "\U0001F4CA Dashboard", "callback_data": "menu_dash"},
        ],
        [
            {"text": "⚙️ Settings", "callback_data": "menu_settings"},
            {"text": "\U0001F4C8 Risk", "callback_data": "menu_risk"},
        ],
        [
            {"text": "\U0001F7E2 Run", "callback_data": "menu_run"},
            {"text": "\U0001F534 Pause", "callback_data": "menu_pause"},
            {"text": "\U0001F6D1 KILL", "callback_data": "menu_kill"},
        ],
    ]}


def kill_confirm_markup() -> dict:
    return {"inline_keyboard": [[
        {"text": "✅ CONFIRM KILL", "callback_data": "menu_killconfirm"},
        {"text": "← Back", "callback_data": "menu_main"},
    ]]}


def persistent_reply_keyboard() -> dict:
    """Always-visible reply keyboard (Telegram is_persistent). Button taps
    arrive as plain text messages, routed by route_keyboard_text()."""
    return {
        "keyboard": [
            [{"text": "\U0001F6D2 Trade"}, {"text": "\U0001F4CA Dashboard"}],
            [{"text": "⚙️ Settings"}, {"text": "\U0001F4CB Menu"}],
        ],
        "is_persistent": True,
        "resize_keyboard": True,
    }


# ── panel text ──

def _pnl_summary(services: ControlServices) -> tuple[str, list[dict]]:
    positions = _safe_positions(services, "BTC")
    if not positions:
        return "Position: none", positions
    p = positions[0]
    return (f"Position: {p.get('positionSide', '?').upper()} {p.get('quantity')} BTC "
            f"@ {p.get('entryPrice')} | mark {p.get('markPrice')} | uPnL {p.get('unrealizedPnl')}"), positions


def trade_panel_text(services: ControlServices) -> str:
    ms = services.store.get_market_state()
    s = services.store.get_strategy_settings()
    state = services.store.get_engine_state()
    pnl_line, _ = _pnl_summary(services)

    lines = ["\U0001F6D2 TRADE PANEL — BTC-PERP",
             f"Engine: {state} | Mode: {s['mode']} | TF: {s['active_bias_tf']}/{s['active_trigger_tf']}"]
    if ms is None:
        lines.append("Market state: not yet published (engine warming up)")
    else:
        lines.append(f"Bias: {ms['bias']} | Last: ${ms['last_price']:,.2f}")
        if ms["long_stop"] and ms["long_target"]:
            rr = (ms["long_target"] - ms["last_price"]) / max(ms["last_price"] - ms["long_stop"], 1e-9)
            lines.append(f"Structural long: stop ${ms['long_stop']:,.0f} / target ${ms['long_target']:,.0f} "
                         f"(R:R {rr:.1f})")
        else:
            lines.append("Structural long: no valid levels — use Custom-Stop Buy")
    lines.append(pnl_line)
    lines.append("Buy presets are RISK budgets (gate-routed, attenuated); sells are reduceOnly.")
    return "\n".join(lines)


# ── buy flows (every path through the gate) ──

async def _manual_buy(update, services: ControlServices, risk_usd: float,
                      stop: float | None = None, target: float | None = None) -> None:
    snap = services.account_snapshot()
    equity = snap["equity"]
    risk_pct = risk_usd / equity
    if not (MIN_RISK_PCT <= risk_pct <= MAX_RISK_PCT):
        lo, hi = equity * MIN_RISK_PCT, equity * MAX_RISK_PCT
        await _reply(update, f"❌ Risk ${risk_usd:,.0f} outside bounds — allowed "
                             f"${lo:,.0f}–${hi:,.0f} at current equity (${equity:,.0f}).")
        return

    ms = services.store.get_market_state()
    if ms is None:
        await _reply(update, "❌ No market state yet — engine hasn't published levels. Try after next cycle.")
        return
    entry = ms["last_price"]

    if stop is None or target is None:  # strategy-anchored default
        stop, target = ms["long_stop"], ms["long_target"]
        if stop is None or target is None:
            await _reply(update, "⚠️ No structural levels for a long right now — pick a custom stop:",
                         reply_markup=custom_stop_markup())
            return

    rr = (target - entry) / max(entry - stop, 1e-9)
    signal = Signal(direction=SignalDirection.LONG, entry=entry, stop=stop, target=target,
                    reward_risk=rr, timestamp=datetime.now(timezone.utc),
                    bias_reason=f"manual buy (bias {ms['bias']})", trigger_reason="trade panel")

    params = services.store.get_risk_params()
    decision = services.gate(
        services.store.get_engine_state(), signal,
        equity=equity, peak_equity=snap["peak_equity"],
        day_start_equity=snap["day_start_equity"],
        open_positions_count=snap["open_positions_count"],
        risk_pct=risk_pct, alpha=params["alpha"], max_concurrent=params["max_concurrent"],
        challenge_cfg=services.store.get_challenge_config(),
        hwm=services.store.get_hwm(),
    )
    if not decision.approved:
        await _reply(update, "❌ Gate rejected:\n- " + "\n- ".join(decision.reasons) +
                             "\n(Manual buys pass the same gate as signals — no bypass.)")
        return

    result = services.execution.create_entry_with_bracket(
        direction="long", quantity=str(decision.quantity),
        stop_trigger=str(stop), target_trigger=str(target), entry_ref_price=str(entry),
    )
    services.store.record_risk_event("manual_buy", {
        "by": _user_id(update), "risk_usd_requested": risk_usd,
        "gated_risk_usd": decision.risk_usd, "attenuation": decision.attenuation_applied,
        "qty": decision.quantity, "dry_run": result.dry_run,
    })
    await _reply(update,
                 f"✅ BUY dispatched (dry_run={result.dry_run})\n"
                 f"qty {decision.quantity} BTC @ ~${entry:,.2f}\n"
                 f"stop ${stop:,.2f} / target ${target:,.2f} (R:R {rr:.1f})\n"
                 f"risk budget ${risk_usd:,.0f} → gated ${decision.risk_usd:,.2f} "
                 f"(attenuation {decision.attenuation_applied:.3f})")


async def cb_trade(update, context, services: ControlServices) -> None:
    """All trd_* callbacks."""
    if _denied(update, "trade"):
        return
    await _ack(update)
    data = update.callback_query.data

    if data == "trd_refresh" or data == "trd_asset_BTC":
        await _reply(update, trade_panel_text(services), reply_markup=trade_panel_markup())
    elif data in ("trd_buy_500", "trd_buy_1000"):
        await _manual_buy(update, services, risk_usd=float(data.rsplit("_", 1)[1]))
    elif data == "trd_buy_custom":
        PENDING_INPUTS[_user_id(update)] = "buy_custom"
        await _reply(update, "✏️ Reply with the risk amount in USD (e.g. 750):")
    elif data == "trd_cstopmenu":
        await _reply(update, "\U0001F3AF Custom-Stop Buy — pick stop distance "
                             f"(target auto-set at {TARGET_MULTIPLE}x distance, default risk %):",
                     reply_markup=custom_stop_markup())
    elif data.startswith("trd_cstop_"):
        d = float(data.rsplit("_", 1)[1]) / 100
        ms = services.store.get_market_state()
        if ms is None:
            await _reply(update, "❌ No market state yet — try after the next engine cycle.")
            return
        entry = ms["last_price"]
        params = services.store.get_risk_params()
        equity = services.account_snapshot()["equity"]
        await _manual_buy(update, services, risk_usd=equity * params["risk_pct"],
                          stop=entry * (1 - d), target=entry * (1 + TARGET_MULTIPLE * d))
    elif data == "trd_sellinit_BTC":
        await _sell_initials(update, services)
    elif data == "trd_sell_custom":
        PENDING_INPUTS[_user_id(update)] = "sell_custom"
        await _reply(update, "✏️ Reply with the percentage to sell (1–100):")
    else:
        await _reply(update, f"Unknown trade action: {data}")


async def _sell_initials(update, services: ControlServices) -> None:
    """Sell back the original cost basis so the remainder rides on profit.
    fraction = entryPrice / markPrice (only meaningful in profit)."""
    positions = _safe_positions(services, "BTC")
    if not positions:
        await _reply(update, "No open BTC position (or no active trading account).")
        return
    p = positions[0]
    entry, mark = float(p["entryPrice"]), float(p.get("markPrice") or 0)
    if mark <= entry:
        await _reply(update, f"⚠️ Sell Initials unavailable — position not in profit "
                             f"(entry ${entry:,.2f}, mark ${mark:,.2f}).")
        return
    fraction = Decimal(str(entry)) / Decimal(str(mark))
    result = services.execution.close_position_market(p, fraction=fraction, purpose="sell_initials")
    await _reply(update, f"✂️ Sell Initials: closing {float(fraction):.1%} "
                         f"(recoups cost basis, dry_run={result.dry_run}).")


# ── custom text-input routing (ForceReply answers + persistent keyboard) ──

KEYBOARD_TEXTS = {"\U0001F6D2 Trade": "trade", "\U0001F4CA Dashboard": "dashboard",
                  "⚙️ Settings": "settings", "\U0001F4CB Menu": "menu"}


async def route_text(update, context, services: ControlServices) -> None:
    """Plain-text messages: persistent-keyboard taps and pending ✏️ inputs."""
    if _denied(update, "text"):
        return
    text = (update.message.text or "").strip()
    uid = _user_id(update)

    if text in KEYBOARD_TEXTS:
        from telegram_control import handlers
        action = KEYBOARD_TEXTS[text]
        if action == "trade":
            await _reply(update, trade_panel_text(services), reply_markup=trade_panel_markup())
        elif action == "dashboard":
            await handlers.cmd_dashboard(update, context, services)
        elif action == "settings":
            await handlers.cmd_settings(update, context, services)
        elif action == "menu":
            await _reply(update, "\U0001F4CB MAIN MENU", reply_markup=main_menu_markup())
        return

    pending = PENDING_INPUTS.pop(uid, None)
    if pending is None:
        return  # unrelated text — ignore
    try:
        value = float(text.replace("$", "").replace("%", "").strip())
    except ValueError:
        await _reply(update, f"❌ Could not parse {text!r} as a number — action cancelled.")
        return

    if pending == "buy_custom":
        await _manual_buy(update, services, risk_usd=value)
    elif pending == "sell_custom":
        if not (1 <= value <= 100):
            await _reply(update, "❌ Percentage must be 1–100 — action cancelled.")
            return
        positions = _safe_positions(services, "BTC")
        if not positions:
            await _reply(update, "No open BTC position (or no active trading account).")
            return
        result = services.execution.close_position_market(
            positions[0], fraction=Decimal(str(value)) / Decimal(100), purpose=f"sell_{value:g}")
        await _reply(update, f"✂️ Sell {value:g}% dispatched (dry_run={result.dry_run}).")


async def cb_trade_command_shim(update, context, services: ControlServices) -> None:
    """/trade command → render the trade panel."""
    if _denied(update, "/trade"):
        return
    await _reply(update, trade_panel_text(services), reply_markup=trade_panel_markup())


# ── main menu ──

async def cmd_menu(update, context, services: ControlServices) -> None:
    if _denied(update, "/menu"):
        return
    # attach the persistent reply keyboard alongside the inline menu
    if getattr(update, "message", None) is not None:
        try:
            from telegram import ReplyKeyboardMarkup
            kb = ReplyKeyboardMarkup.de_json(persistent_reply_keyboard(), None)
            await update.message.reply_text("\U0001F4CB MAIN MENU — persistent keyboard enabled",
                                            reply_markup=kb)
        except ImportError:
            pass
    await _reply(update, "Key functions:", reply_markup=main_menu_markup())


async def cb_menu(update, context, services: ControlServices) -> None:
    """All menu_* callbacks — routes into the existing command handlers."""
    if _denied(update, "menu"):
        return
    await _ack(update)
    from telegram_control import handlers
    data = update.callback_query.data

    if data == "menu_main":
        await _reply(update, "\U0001F4CB MAIN MENU", reply_markup=main_menu_markup())
    elif data == "menu_trade":
        await _reply(update, trade_panel_text(services), reply_markup=trade_panel_markup())
    elif data == "menu_dash":
        await handlers.cmd_dashboard(update, context, services)
    elif data == "menu_settings":
        s = services.store.get_strategy_settings()
        await _reply(update, handlers._settings_text(s, services.store),
                     reply_markup=handlers.settings_menu_markup(s))
    elif data == "menu_risk":
        await handlers.cmd_risk(update, context, services)
    elif data == "menu_run":
        await handlers.cmd_run(update, context, services)
    elif data == "menu_pause":
        await handlers.cmd_pause(update, context, services)
    elif data == "menu_kill":
        await _reply(update, "\U0001F6D1 Kill switch: cancel ALL orders and close ALL positions?",
                     reply_markup=kill_confirm_markup())
    elif data == "menu_killconfirm":
        await handlers.cmd_kill(update, context, services)
    else:
        await _reply(update, f"Unknown menu action: {data}")
