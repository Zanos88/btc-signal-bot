-- 2026-09-02 RLS hardening step 1 of 3 — enforce_floor_guard must FAIL CLOSED on a missing equity_hwm row.
-- Why first: the function is SECURITY INVOKER. Under RLS a non-BYPASSRLS caller sees ZERO rows from equity_hwm,
-- hwm_val stays NULL and COALESCE(hwm_val, cfg.initial_balance) silently drops the trailing floor to
-- initial_balance — while challenge_config two lines above already fails closed and loud. Enabling RLS before this
-- patch ARMS that hole. This is the only behavioural change: a missing/unreadable hwm row now REJECTS entry intents
-- (risk-reducing intents are untouched — they return NEW before any lookup).
-- Also pins `search_path = public` (Supabase advisor WARN function_search_path_mutable on both projects).
-- Idempotent (CREATE OR REPLACE). Rollback: re-run the function body from db/schema.sql (the COALESCE form).
CREATE OR REPLACE FUNCTION enforce_floor_guard() RETURNS TRIGGER AS $$
DECLARE
    latest RECORD;
    cfg RECORD;
    hwm_val NUMERIC;
    dd_base NUMERIC;
    dd_floor NUMERIC;
    worst_case NUMERIC;
    floor_binding NUMERIC;
BEGIN
    IF NEW.reduce_only OR NEW.close_position OR NEW.purpose <> 'entry' THEN
        RETURN NEW;
    END IF;

    IF NEW.risk_entry_price IS NULL OR NEW.risk_stop_price IS NULL THEN
        RAISE EXCEPTION 'floor guard: entry intent % missing risk_entry_price/risk_stop_price (fail closed)',
            NEW.intent_id;
    END IF;

    SELECT equity, day_start_equity INTO latest
    FROM portfolio_telemetry
    ORDER BY ts DESC, id DESC
    LIMIT 1;

    IF latest IS NULL THEN
        RAISE EXCEPTION 'floor guard: no telemetry rows — refusing entry intent % (fail closed)',
            NEW.intent_id;
    END IF;

    SELECT drawdown_type, max_drawdown_pct, daily_loss_pct, initial_balance
        INTO cfg FROM challenge_config WHERE id = 1;
    IF cfg IS NULL THEN
        RAISE EXCEPTION 'floor guard: challenge_config missing — refusing entry intent % (fail closed)',
            NEW.intent_id;
    END IF;

    SELECT hwm INTO hwm_val FROM equity_hwm WHERE id = 1;
    -- 2026-09-02: FAIL CLOSED. A trailing-drawdown tier with no readable high-water mark cannot compute its floor;
    -- refusing the entry is the only safe answer (under RLS an unprivileged caller reads zero rows here).
    IF NOT FOUND OR hwm_val IS NULL THEN
        IF cfg.drawdown_type = 'trailing' THEN
            RAISE EXCEPTION 'floor guard: equity_hwm row missing/unreadable — refusing entry intent % (fail closed)',
                NEW.intent_id;
        END IF;
        hwm_val := cfg.initial_balance;   -- static tiers never use the hwm; keep the historical value
    END IF;

    dd_base := CASE WHEN cfg.drawdown_type = 'trailing'
                    THEN GREATEST(hwm_val, cfg.initial_balance)
                    ELSE cfg.initial_balance END;
    dd_floor := dd_base * (1 - cfg.max_drawdown_pct / 100);
    floor_binding := GREATEST(
        latest.day_start_equity - cfg.initial_balance * cfg.daily_loss_pct / 100,
        dd_floor);
    worst_case := latest.equity - ABS(NEW.risk_entry_price - NEW.risk_stop_price) * NEW.quantity;

    IF worst_case <= floor_binding + 200 THEN
        RAISE EXCEPTION 'floor guard: intent % worst-case equity % crosses binding floor % + 200 buffer',
            NEW.intent_id, round(worst_case, 2), floor_binding;
    END IF;

    RETURN NEW;
END
$$ LANGUAGE plpgsql SET search_path = public;   -- 2026-09-02: pins search_path (advisor function_search_path_mutable)
