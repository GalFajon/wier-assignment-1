SELECT
  (SELECT COUNT(*) FROM public.site) AS site_count,
  (SELECT COUNT(*) FROM public.page) AS page_count,
  (SELECT COUNT(*) FROM public.page WHERE page_type_code = 'DUPLICATE') AS duplicate_count,
  (SELECT COUNT(*) FROM public.image) AS image_count,
  (
    SELECT COALESCE(AVG(img_cnt), 0)
    FROM (
      SELECT p.id, COUNT(i.id) AS img_cnt
      FROM public.page p
      LEFT JOIN public.image i ON i.page_id = p.id
      WHERE p.page_type_code = 'HTML'
      GROUP BY p.id
    ) t
  ) AS avg_images_per_page,
  (
    SELECT COALESCE(AVG(link_cnt), 0)
    FROM (
      SELECT p.id, COUNT(l.to_page) AS link_cnt
      FROM public.page p
      LEFT JOIN public.link l ON l.from_page = p.id
      GROUP BY p.id
    ) t
  ) AS avg_links_per_page
;

-- Pages with the most links (outgoing)
WITH link_counts AS (
  SELECT p.id, p.url, COUNT(l.to_page) AS link_count
  FROM public.page p
  LEFT JOIN public.link l ON l.from_page = p.id
  GROUP BY p.id, p.url
)
SELECT id, url, link_count
FROM link_counts
ORDER BY link_count DESC;

-- Pages with the most images
WITH image_counts AS (
  SELECT p.id, p.url, COUNT(i.id) AS image_count
  FROM public.page p
  LEFT JOIN public.image i ON i.page_id = p.id
  GROUP BY p.id, p.url
)
SELECT id, url, image_count
FROM image_counts
ORDER BY image_count DESC;

SELECT
  pd.data_type_code,
  COUNT(*) AS binary_doc_count
FROM public.page_data pd
GROUP BY pd.data_type_code
ORDER BY pd.data_type_code;