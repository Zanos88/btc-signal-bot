-- 2026-09-02 RLS hardening step 3 of 3 — enable Row Level Security on every public table (no policies = deny for
-- non-BYPASSRLS roles; the bot's postgres-role connection is unaffected). Run ONLY after 01 (guard fails closed)
-- and 02 (grants revoked). Idempotent. Clears the Supabase advisor `rls_disabled_in_public`.
BEGIN;
DO $$
DECLARE t RECORD;
BEGIN
  FOR t IN SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public' AND NOT rowsecurity LOOP
    EXECUTE format('ALTER TABLE %I.%I ENABLE ROW LEVEL SECURITY', t.schemaname, t.tablename);
  END LOOP;
END $$;
COMMIT;

-- VERIFY (expect zero rows)
SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND NOT rowsecurity;
-- and (expect: no policies, force_rls = false everywhere)
SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, (SELECT count(*) FROM pg_policies p WHERE p.tablename = c.relname) AS policies
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relkind = 'r' ORDER BY 1;
