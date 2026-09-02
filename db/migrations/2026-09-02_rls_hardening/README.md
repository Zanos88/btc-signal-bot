# 2026-09-02 Supabase RLS hardening — btc-signal-bot (prod `lnycymeylmhjqpwtdint`, staging `bgkddwsnnawczecixsuf`)

**Trigger:** Supabase advisor `rls_disabled_in_public` on both projects (email 31 Aug 2026) + an audit that proved the anon key can `PATCH /rest/v1/equity_hwm` from the internet (the high-water mark the live drawdown floor is computed from).

**Order matters (run from the SQL editor, staging first, then prod):**
1. `01_floor_guard_fail_closed.sql` — `enforce_floor_guard` refuses entries when `equity_hwm` is missing/unreadable on a trailing tier (it used to fail OPEN to `initial_balance`; enabling RLS first would have armed that).
2. `02_revoke_anon_grants.sql` — revoke anon/authenticated write on all public tables + sequences, and revoke the default ACL so new tables don't inherit `arwdDxtm`. The bot and the Vercel API use `DATABASE_URL` (postgres role), not PostgREST roles → unaffected.
3. `03_enable_rls.sql` — enable RLS everywhere (no policies). Clears the advisor.

**Verify after each project:** the SELECTs at the bottom of 02 and 03 return zero rows; Supabase → Advisors shows 0 `rls_disabled_in_public`; the bot logs a heartbeat / telemetry row within 10 min; a PATCH with the anon key against **staging only** returns 401/403 or `[]`.

**Process rule (recorded):** write-probes are never run against production. Staging only, or a transaction that is rolled back.

**Rollback:** `99_rollback.sql` (RLS off). Grants are deliberately not restored.
