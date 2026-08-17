"""Unit tests for alerts/telegram.py::TelegramClient.

The message *formatters* are covered by test_alert_formats/test_severity_tiers,
but the delivery wrapper itself (config guard, payload assembly, and the
documented "never a silent except" failure contract) had no coverage. These
tests use a fake `requests.post` so nothing hits the network.
"""
from __future__ import annotations

import pytest
import requests

import alerts.telegram as tg
from alerts.telegram import TelegramClient, TelegramConfigError


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("BTC_SIGNAL_BOT_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("BTC_SIGNAL_BOT_TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(TelegramConfigError):
        TelegramClient()


class _FakeResponse:
    def __init__(self, ok: bool):
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.HTTPError("boom")


def test_send_success_builds_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(ok=True)

    monkeypatch.setattr(tg.requests, "post", fake_post)
    client = TelegramClient(bot_token="T", chat_id="C")

    ok = client.send("hello", reply_markup={"inline_keyboard": []}, silent=True)

    assert ok is True
    assert captured["url"].endswith("/botT/sendMessage")
    body = captured["json"]
    assert body["chat_id"] == "C" and body["text"] == "hello"
    assert body["parse_mode"] == "HTML"          # default
    assert body["disable_notification"] is True   # silent=True
    assert body["reply_markup"] == {"inline_keyboard": []}


def test_send_omits_parse_mode_when_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(tg.requests, "post",
                        lambda url, json=None, timeout=None: captured.update(json=json) or _FakeResponse(True))
    TelegramClient(bot_token="T", chat_id="C").send("x", parse_mode=None)
    assert "parse_mode" not in captured["json"]
    assert "disable_notification" not in captured["json"]  # silent defaults False


def test_send_failure_returns_false_not_raises(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(tg.requests, "post", fake_post)
    client = TelegramClient(bot_token="T", chat_id="C")

    # Contract: a swallowed delivery failure would defeat the heartbeat, so
    # send() logs a WARNING and returns False rather than raising.
    assert client.send("hello") is False
