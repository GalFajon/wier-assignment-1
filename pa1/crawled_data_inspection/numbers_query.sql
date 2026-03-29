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
      GROUP BY p.id
    ) t
  ) AS avg_images_per_page
;

SELECT
  pd.data_type_code,
  COUNT(*) AS binary_doc_count
FROM public.page_data pd
GROUP BY pd.data_type_code
ORDER BY pd.data_type_code;