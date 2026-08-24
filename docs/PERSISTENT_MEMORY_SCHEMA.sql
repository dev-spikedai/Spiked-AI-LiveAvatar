-- Durable cognitive-twin memory. Deploy this in the application's Supabase
-- project before enabling cross-session memory writes.
--
-- The orchestrator treats this table as optional: until it exists, meetings
-- continue with working and ephemeral memory only.

create table if not exists public.agent_memory (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  client_id text not null default '',
  memory_type text not null check (memory_type in ('preference', 'role', 'fact', 'decision', 'terminology')),
  memory_key text not null,
  memory_value jsonb not null,
  confidence numeric not null default 0.9 check (confidence >= 0 and confidence <= 1),
  source text not null default 'meeting_instruction',
  expires_at timestamptz null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, client_id, memory_type, memory_key)
);

create index if not exists agent_memory_user_scope_idx
  on public.agent_memory (user_id, client_id, updated_at desc);

alter table public.agent_memory enable row level security;

-- Service-role access is used by the orchestrator. If browser-side access is
-- later needed, add explicit policies; do not make this table public.
