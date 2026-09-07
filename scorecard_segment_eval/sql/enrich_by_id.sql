-- Package default of sql/enrich_by_id.sql (override with a repo-level copy).
SELECT
    e.SKP_CREDIT_CASE      AS col_id,
    e.DTIME_SCORE          AS col_date,
    e.CODE_CHANNEL         AS segment_channel,
    e.CODE_PRODUCT         AS segment_product,
    e.TARGET_DEFAULT       AS col_target,
    e.TARGET_DEFAULT_OBS   AS col_obs
FROM APP_ENRICH e
WHERE e.SKP_CREDIT_CASE IN ({id_list})
