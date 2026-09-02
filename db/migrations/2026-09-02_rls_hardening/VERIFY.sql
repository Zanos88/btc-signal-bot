-- Run AFTER ALL_IN_ONE.sql. Run each statement separately (the editor shows one result at a time).
-- 1) EXPECT ZERO ROWS: leftover anon/authenticated write grants
SELECT grantee, table_name, privilege_type FROM information_schema.role_table_grants
 WHERE table_schema = 'public' AND grantee IN ('anon','authenticated')
   AND privilege_type IN ('INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER')
 ORDER BY table_name, grantee;

-- 2) EXPECT ZERO ROWS: tables still without RLS
SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND NOT rowsecurity;

-- 3) EXPECT: every table relrowsecurity = true, policies = 0, relforcerowsecurity = false
SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, (SELECT count(*) FROM pg_policies p WHERE p.tablename = c.relname) AS policies
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relkind = 'r' ORDER BY 1;
