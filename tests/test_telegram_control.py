"""Phase 5 acceptance tests: control-plane handlers with fake services.

Proves: auth rejection (fail closed), /kill from any state, /run confirm
required from KILLED, /risk bounds enforcement, and that the Take button
routes through the risk gate (and respects its rejection) — no
fixed-notional path exists anywhere.
"""
from __future__ import annotations

import asyncio

import pytest

from risk.gate import GateDecision
from telegram_control import handlers
from telegram_control.handlers import ControlServices

ADMIN_ID = 111111
INTRUDER_ID = 999999


# ── fakes ──

class FakeMessage:
    def __init__(self):
        self.replies: list[str] = []
        self.markups: list = []

    async def reply_text(self, text, reply_markup=None, parse_mode=None):
        self.replies.append(text)
        self.markups.append(reply_markup)


class FakeCallbackQuery:
    def __init__(self, data):
        self.data = data
        self.answered = False
        self.edits: list[str] = []
        self.markups: list = []

    async def answer(self):
        self.answered = True

    async def edit_message_text(self, text, reply_markup=None, parse_mode=None):
        self.edits.append(text)
        self.markups.append(reply_markup)


class FakeUpdate:
    def __init__(self, user_id, callback_data=None):
        self.effective_user = type("U", (), {"id": user_id})()
        self.message = FakeMessage() if callback_data is None else None
        self.callback_query = FakeCallbackQuery(callback_data) if callback_data else None

    def output(self) -> str:
        if self.message is not None:
            return "\n".join(self.message.replies)
        return "\n".join(self.callback_query.edits)


class FakeContext:
    def __init__(self, args=None):
        self.args = args or []


class FakeStore:
    def __init__(self, state="ACTIVE"):
        self.state = state
        self.events = []
        self.params = {"risk_pct": 0.0075, "alpha": 1.5, "max_concurrent": 1}
        self.pending = {}

    def get_challenge_config(self):
        from risk.challenge import DEFAULT_CONFIG
        return DEFAULT_CONFIG

    def get_hwm(self):
        return 100_000.0

    def get_engine_state(self):
        return self.state

    def set_engine_state(self, state, updated_by):
        self.state = state

    def record_risk_event(self, event_type, detail=None):
        self.events.append((event_type, detail or {}))

    def get_risk_params(self):
        return dict(self.params)

    def set_risk_param(self, name, value, updated_by):
        if name == "risk_pct" and not (0.0025 <= value <= 0.01):
            raise ValueError("risk_pct outside bounds")
        old = self.params[name]
        self.params[name] = value
        self.events.append(("risk_param_change", {"param": name, "old": old, "new": value}))
        return dict(self.params)

    def get_pending_signal(self, signal_id):
        return self.pending.get(signal_id)

    def resolve_pending_signal(self, signal_id, status, resolved_by):
        if signal_id in self.pending:
            self.pending[signal_id]["status"] = status


class FakeExecution:
    def __init__(self):
        self.kill_calls = 0
        self.entries = []
        self.closes = []
        self.positions = []

    def kill_sequence(self):
        self.kill_calls += 1
        return {"dry_run": True, "cancelled_order_ids": ["o1"], "closed": ["p1"]}

    def create_entry_with_bracket(self, **kwargs):
        self.entries.append(kwargs)
        return type("R", (), {"dry_run": True})()

    def get_open_positions(self, base=None):
        return self.positions

    def close_position_market(self, pos, fraction, purpose):
        self.closes.append((pos, fraction, purpose))
        return type("R", (), {"dry_run": True})()


def make_services(state="ACTIVE", gate=None):
    store, execution = FakeStore(state), FakeExecution()
    snapshot = lambda: {"equity": 100_000, "peak_equity": 100_000,
                        "day_start_equity": 100_000, "open_positions_count": 0}
    services = ControlServices(store=store, execution=execution,
                               account_snapshot=snapshot, gate=gate or handlers.evaluate_gate)
    return services, store, execution


@pytest.fixture(autouse=True)
def admin_allowlist(monkeypatch):
    monkeypatch.setenv("BTC_SIGNAL_BOT_ADMIN_IDS", str(ADMIN_ID))


def run(coro):
    return asyncio.run(coro)


# ── auth ──

def test_unauthorized_user_is_ignored_everywhere():
    services, store, execution = make_services()
    upd = FakeUpdate(INTRUDER_ID)
    run(handlers.cmd_kill(upd, FakeContext(), services))
    run(handlers.cmd_run(upd, FakeContext(), services))
    assert execution.kill_calls == 0
    assert store.state == "ACTIVE"
    assert upd.output() == ""                       # silence, no state leak


def test_empty_allowlist_fails_closed(monkeypatch):
    monkeypatch.setenv("BTC_SIGNAL_BOT_ADMIN_IDS", "")
    services, _, execution = make_services()
    upd = FakeUpdate(ADMIN_ID)                       # even the owner is locked out
    run(handlers.cmd_kill(upd, FakeContext(), services))
    assert execution.kill_calls == 0


# ── /kill and /run confirm ──

def test_kill_works_from_any_state_and_locks():
    for state in ("ACTIVE", "PAUSED", "KILLED"):
        services, store, execution = make_services(state)
        upd = FakeUpdate(ADMIN_ID)
        run(handlers.cmd_kill(upd, FakeContext(), services))
        assert execution.kill_calls == 1
        assert store.state == "KILLED"
        assert ("kill_invoked" in [e for e, _ in store.events])
        assert "KILL executed" in upd.output()


def test_run_from_killed_requires_confirm():
    services, store, _ = make_services("KILLED")
    upd = FakeUpdate(ADMIN_ID)
    run(handlers.cmd_run(upd, FakeContext(), services))
    assert store.state == "KILLED"                   # refused without confirm
    assert "confirm" in upd.output()

    upd2 = FakeUpdate(ADMIN_ID)
    run(handlers.cmd_run(upd2, FakeContext(args=["confirm"]), services))
    assert store.state == "ACTIVE"


def test_pause_sets_paused():
    services, store, _ = make_services("ACTIVE")
    run(handlers.cmd_pause(FakeUpdate(ADMIN_ID), FakeContext(), services))
    assert store.state == "PAUSED"


# ── /risk ──

def test_risk_bounds_enforced_and_logged():
    services, store, _ = make_services()
    upd = FakeUpdate(ADMIN_ID)
    run(handlers.cmd_risk(upd, FakeContext(args=["pct", "0.5"]), services))
    assert store.params["risk_pct"] == pytest.approx(0.005)
    assert ("risk_param_change" in [e for e, _ in store.events])

    upd2 = FakeUpdate(ADMIN_ID)
    run(handlers.cmd_risk(upd2, FakeContext(args=["pct", "0.2"]), services))
    assert store.params["risk_pct"] == pytest.approx(0.005)     # unchanged
    assert "rejected" in upd2.output()


# ── Frame A: take routes through the gate ──

def test_take_button_routes_through_gate_and_respects_rejection():
    calls = []

    def rejecting_gate(*args, **kwargs):
        calls.append(kwargs)
        return GateDecision(approved=False, reasons=["test rejection"])

    services, store, execution = make_services(gate=rejecting_gate)
    store.pending["SIG1"] = {"signal_id": "SIG1", "direction": "LONG", "entry": 60_000.0,
                             "stop": 59_000.0, "target": 62_500.0, "reward_risk": 2.5,
                             "status": "pending"}
    upd = FakeUpdate(ADMIN_ID, callback_data="take_0.75_SIG1")
    run(handlers.cb_take_signal(upd, FakeContext(), services))

    assert len(calls) == 1                            # gate WAS consulted
    assert calls[0]["risk_pct"] == pytest.approx(0.0075)
    assert execution.entries == []                    # rejection honored — nothing dispatched
    assert "Gate rejected" in upd.output()


def test_take_button_dispatches_gate_quantity_on_approval():
    def approving_gate(*args, **kwargs):
        return GateDecision(approved=True, quantity=0.42, risk_usd=420.0, attenuation_applied=1.0)

    services, store, execution = make_services(gate=approving_gate)
    store.pending["SIG2"] = {"signal_id": "SIG2", "direction": "LONG", "entry": 60_000.0,
                             "stop": 59_000.0, "target": 62_500.0, "reward_risk": 2.5,
                             "status": "pending"}
    upd = FakeUpdate(ADMIN_ID, callback_data="take_0.5_SIG2")
    run(handlers.cb_take_signal(upd, FakeContext(), services))

    assert len(execution.entries) == 1
    entry = execution.entries[0]
    assert entry["quantity"] == "0.42"                # gate's sized quantity, not fixed notional
    assert entry["stop_trigger"] == "59000.0"
    assert store.pending["SIG2"]["status"] == "taken"


def test_stale_signal_cannot_be_taken():
    services, store, execution = make_services()
    store.pending["SIG3"] = {"signal_id": "SIG3", "direction": "LONG", "entry": 60_000.0,
                             "stop": 59_000.0, "target": 62_500.0, "reward_risk": 2.5,
                             "status": "taken"}
    upd = FakeUpdate(ADMIN_ID, callback_data="take_0.75_SIG3")
    run(handlers.cb_take_signal(upd, FakeContext(), services))
    assert execution.entries == []
    assert "expired or already resolved" in upd.output()


# ── Frame B ──

def test_close_fraction_button():
    services, _, execution = make_services()
    execution.positions = [{"positionId": "P1", "positionSide": "long", "quantity": "0.5", "base": "BTC"}]
    upd = FakeUpdate(ADMIN_ID, callback_data="close_50_BTC")
    run(handlers.cb_close_fraction(upd, FakeContext(), services))
    assert len(execution.closes) == 1
    _, fraction, purpose = execution.closes[0]
    assert float(fraction) == 0.5 and purpose == "close_50"


def test_close_button_survives_no_active_account():
    """Regression (found in live Frame B testing): with no active challenge
    the SDK raises `account_id not set` — the handler must reply
    gracefully, not crash unanswered."""
    services, _, execution = make_services()

    def raising(base=None):
        raise ValueError("account_id not set. Call client.setup() first")

    execution.get_open_positions = raising
    upd = FakeUpdate(ADMIN_ID, callback_data="close_50_BTC")
    run(handlers.cb_close_fraction(upd, FakeContext(), services))
    assert "No open BTC position" in upd.output()      # graceful reply, no crash

    upd2 = FakeUpdate(ADMIN_ID, callback_data="slbe_BTC")
    run(handlers.cb_sl_breakeven(upd2, FakeContext(), services))
    assert "No open BTC position" in upd2.output()


def test_frame_b_markup_shape():
    markup = handlers.frame_b_markup("BTC")
    rows = markup["inline_keyboard"]
    all_callbacks = [b["callback_data"] for row in rows for b in row]
    assert all_callbacks == ["close_25_BTC", "close_50_BTC", "close_100_BTC", "slbe_BTC"]
    assert all(len(cb.encode()) <= 64 for cb in all_callbacks)   # Telegram callback_data limit


def test_frame_a_markup_round_trips_through_take_skip_handlers():
    """Producer/consumer contract: the Frame A keyboard built in
    main.frame_a_markup must emit callback_data that the take/skip handlers
    parse back correctly. Both sides were previously tested only in
    isolation (handlers against hand-written strings), so a change to the
    `take_<pct>_<id>` / `skip_<id>` format would slip through silently.
    """
    from main import frame_a_markup

    signal_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"  # 26-char Crockford ULID (no underscores)
    rows = frame_a_markup(signal_id)["inline_keyboard"]
    callbacks = [b["callback_data"] for row in rows for b in row]

    assert callbacks == [f"take_0.75_{signal_id}", f"take_0.5_{signal_id}", f"skip_{signal_id}"]
    assert all(len(cb.encode()) <= 64 for cb in callbacks)  # Telegram callback_data limit

    # cb_take_signal parses with split("_", 2) -> [verb, pct, signal_id]
    for cb in callbacks[:2]:
        verb, pct_str, parsed_id = cb.split("_", 2)
        assert verb == "take"
        assert float(pct_str) in (0.75, 0.5)
        assert parsed_id == signal_id

    # cb_skip_signal parses with split("_", 1) -> [verb, signal_id]
    verb, parsed_id = callbacks[2].split("_", 1)
    assert verb == "skip"
    assert parsed_id == signal_id


def test_dashboard_attaches_frame_b_only_with_positions():
    # with a position -> Frame B keyboard attached
    services, _, execution = make_services()
    execution.positions = [{"positionId": "P1", "positionSide": "long", "quantity": "0.5",
                            "base": "BTC", "entryPrice": "60000", "markPrice": "60100",
                            "unrealizedPnl": "50"}]
    upd = FakeUpdate(ADMIN_ID)
    run(handlers.cmd_dashboard(upd, FakeContext(), services))
    assert upd.message.markups[-1] is not None

    # without positions -> plain text, no keyboard
    services2, _, execution2 = make_services()
    execution2.positions = []
    upd2 = FakeUpdate(ADMIN_ID)
    run(handlers.cmd_dashboard(upd2, FakeContext(), services2))
    assert upd2.message.markups[-1] is None
