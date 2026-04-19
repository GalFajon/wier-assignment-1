CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE public.page ADD COLUMN IF NOT EXISTS cleaned_content TEXT;

CREATE TABLE IF NOT EXISTS public.page_segment (
  id serial NOT NULL,
  page_id integer,
  page_segment TEXT,
  embedding vector(768)
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'fk_page_page_segment'
  ) THEN
    ALTER TABLE public.page_segment
      ADD CONSTRAINT fk_page_page_segment
      FOREIGN KEY (page_id)
      REFERENCES public.page(id)
      ON DELETE RESTRICT;
  END IF;
END $$;