-- Enrich optional application attributes by credit-case id.
-- Replace schema/table names for your warehouse.
-- Placeholders:
--   {id_list}  comma-separated quoted identifiers from the base table
--   {col_id}   id column name (default SKP_CREDIT_CASE)
--
-- The SELECT aliases below are recognized automatically:
--   col_id, col_date, col_target, col_obs, segment_*

SELECT
    e.SKP_CREDIT_CASE      AS col_id,
    e.DTIME_SCORE          AS col_date,
    e.CODE_CHANNEL         AS segment_channel,
    e.CODE_PRODUCT         AS segment_product,
    e.TARGET_DEFAULT       AS col_target,
    e.TARGET_DEFAULT_OBS   AS col_obs
FROM APP_ENRICH e
WHERE e.SKP_CREDIT_CASE IN ({id_list})
