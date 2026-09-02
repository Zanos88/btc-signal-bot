-- ROLLBACK (only if the bot or the API breaks). Order: RLS off first (instant relief), grants are NOT restored
-- (they were the vulnerability), guard stays fail-closed unless the trailing tier itself is broken.
-- If you use the Supabase migration ledger, ALSO delete the ledger rows for 01/02/03 or the next deploy replays them.
DO $$
DECLARE t RECORD;
BEGIN
  FOR t IN SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public' AND rowsecurity LOOP
    EXECUTE format('ALTER TABLE %I.%I DISABLE ROW LEVEL SECURITY', t.schemaname, t.tablename);
  END LOOP;
END $$;
-- Guard rollback (ONLY if trailing-tier entries are being refused because equity_hwm is legitimately absent):
--   re-run the enforce_floor_guard body from db/schema.sql (COALESCE form) — then FIX the missing hwm row instead.
