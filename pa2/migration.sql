CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE public.page
ADD COLUMN IF NOT EXISTS cleaned_content TEXT;

CREATE TABLE IF NOT EXISTS public.model (
  id serial PRIMARY KEY,
  model_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS public.page_segment_vec384 (
  id serial PRIMARY KEY,
  page_id integer REFERENCES public.page(id) ON DELETE RESTRICT,
  model_id integer REFERENCES public.model(id) ON DELETE RESTRICT,
  page_segment TEXT,
  embedding vector(384)
);

CREATE TABLE IF NOT EXISTS public.page_segment_vec768 (
  id serial PRIMARY KEY,
  page_id integer REFERENCES public.page(id) ON DELETE RESTRICT,
  model_id integer REFERENCES public.model(id) ON DELETE RESTRICT,
  page_segment TEXT,
  embedding vector(768)
);

CREATE TABLE IF NOT EXISTS public.page_segment_vec1024 (
  id serial PRIMARY KEY,
  page_id integer REFERENCES public.page(id) ON DELETE RESTRICT,
  model_id integer REFERENCES public.model(id) ON DELETE RESTRICT,
  page_segment TEXT,
  embedding vector(1024)
);