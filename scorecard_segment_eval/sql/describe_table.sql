-- query: describe_table
-- Column catalogue of the analysis table, used to discover which optional
-- columns (date, segmentation, target, observation flag) actually exist.
-- Parameters: table, table_schema, table_name
SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE lower(table_name) = lower('{table_name}')
  AND ('{table_schema}' = '' OR lower(table_schema) = lower('{table_schema}'))
ORDER BY ordinal_position
