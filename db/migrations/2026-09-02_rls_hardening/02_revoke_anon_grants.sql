-- 2026-09-02 RLS hardening step 2 of 3 — the DURABLE fix: strip anon/authenticated write grants and stop new tables
-- from inheriting them. Audit finding: anon + authenticated hold DELETE/INSERT/UPDATE/TRUNCATE/SELECT on every
-- public table (pg_default_acl grants arwdDxtm to anon on each new table); a live anon PATCH /rest/v1/equity_hwm
-- succeeded from the open internet. RLS cannot filter TRUNCATE, and one future CREATE POLICY ... USING (true)
-- would restore full anon write — grants are the real boundary.
-- The bot (db/store.py) and the Vercel API (web/api/index.py) connect with DATABASE_URL (postgres role via the
-- session pooler), NOT the anon/authenticated PostgREST roles, so they are unaffected. Run in ONE transaction.
BEGIN;
DO $$
DECLARE t RECORD;
BEGIN
  FOR t IN SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public' LOOP
    EXECUTE format('REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON TABLE %I.%I FROM anon, authenticated',
                   t.schemaname, t.tablename);
  END LOOP;
END $$;
-- Sequences: anon must not be able to advance/reset them either.
DO $$
DECLARE s RECORD;
BEGIN
  FOR s IN SELECT sequence_schema, sequence_name FROM information_schema.sequences WHERE sequence_schema = 'public' LOOP
    EXECUTE format('REVOKE ALL ON SEQUENCE %I.%I FROM anon, authenticated', s.sequence_schema, s.sequence_name);
  END LOOP;
END $$;
-- Default privileges: stop FUTURE tables/sequences inheriting anon write. Supabase grants defaults from both
-- `postgres` and `supabase_admin`; revoke for both (a missing role simply errors — tolerated below).
DO $$
BEGIN
  BEGIN
    EXECUTE 'ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON TABLES FROM anon, authenticated';
    EXECUTE 'ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON SEQUENCES FROM anon, authenticated';
  EXCEPTION WHEN OTHERS THEN RAISE NOTICE 'default-privilege revoke for postgres skipped: %', SQLERRM; END;
  BEGIN
    EXECUTE 'ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON TABLES FROM anon, authenticated';
    EXECUTE 'ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public REVOKE ALL ON SEQUENCES FROM anon, authenticated';
  EXCEPTION WHEN OTHERS THEN RAISE NOTICE 'default-privilege revoke for supabase_admin skipped: %', SQLERRM; END;
END $$;
COMMIT;

-- VERIFY (expect zero rows): any remaining anon/authenticated write privilege on public tables
SELECT grantee, table_name, privilege_type FROM information_schema.role_table_grants
 WHERE table_schema = 'public' AND grantee IN ('anon','authenticated')
   AND privilege_type IN ('INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER')
 ORDER BY table_name, grantee;
