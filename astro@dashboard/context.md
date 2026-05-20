## I have set the environment variable
NEXT_PUBLIC_SUPABASE_URL
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY

## Codes run in supabase

```
create table if not exists public.reports (
  id text primary key,
  patient_name text,
  created_at timestamptz,
  filename text,
  status text,
  diagnosis text,
  infection_rate numeric,
  model_confidence numeric,
  inference_mode text,
  model_probability numeric,
  total_cells integer,
  infected_cells integer,
  healthy_cells integer,
  quality_ok boolean,
  quality_message text
);

alter table public.reports enable row level security;

create policy "allow read reports"
on public.reports for select
using (true);

create policy "allow insert reports"
on public.reports for insert
with check (true);
```

```
notify pgrst, 'reload schema';
```

```
alter table public.reports
  add column if not exists dob date,
  add column if not exists blood_pressure text,
  add column if not exists blood_o2 numeric;
 ```

