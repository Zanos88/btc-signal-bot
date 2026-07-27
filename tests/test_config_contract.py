"""Guard the config.yaml <-> code-constant contract.

Several `config.yaml` keys are documentation of the code's live defaults,
not runtime inputs (see README "### Configuring"): the engine uses module
constants / the DB `risk_params`, never these keys. That makes them prone
to silent drift — someone edits the YAML expecting an effect, or bumps a
code constant and forgets the doc. These tests pin the two together so the
documented seed can never quietly diverge from the value the engine
actually uses.

Pure/read-only: loads the committed config.yaml and imports the constants;
no DB, no network, no live API.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_risk_pct_seed_matches_sizing_default():
    from risk.sizing import DEFAULT_RISK_PCT

    assert _config()["risk"]["risk_pct"] == DEFAULT_RISK_PCT


def test_btc_sz_decimals_seed_matches_sizing_default():
    from risk.sizing import DEFAULT_BTC_SZ_DECIMALS

    assert _config()["risk"]["btc_sz_decimals"] == DEFAULT_BTC_SZ_DECIMALS


def test_circuit_breaker_halt_pct_seed_matches_hardcoded_threshold():
    from risk.circuit_breaker import HALT_THRESHOLD_PCT

    assert _config()["risk"]["circuit_breaker_halt_pct"] == HALT_THRESHOLD_PCT


def test_indicator_period_seeds_match_module_constants():
    from strategy.trigger_1h import FISHER_PERIOD, OBV_SMA_PERIOD

    strat = _config()["strategy"]
    assert strat["fisher_period"] == FISHER_PERIOD
    assert strat["obv_sma_period"] == OBV_SMA_PERIOD


def test_min_reward_risk_seed_matches_module_constants():
    from risk.gate import MIN_REWARD_RISK as GATE_MIN_RR
    from strategy.signals import MIN_REWARD_RISK as SIGNAL_MIN_RR

    # The two constants must agree with each other and with the seed.
    assert GATE_MIN_RR == SIGNAL_MIN_RR
    assert _config()["strategy"]["min_reward_risk"] == SIGNAL_MIN_RR


def test_bias_param_seeds_match_compute_bias_defaults():
    from strategy.bias_4h import compute_bias

    defaults = inspect.signature(compute_bias).parameters
    strat = _config()["strategy"]
    assert strat["fractal_width"] == defaults["fractal_width"].default
    assert strat["sr_lookback"] == defaults["sr_lookback"].default
