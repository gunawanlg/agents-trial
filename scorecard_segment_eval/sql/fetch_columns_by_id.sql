-- query: fetch_columns_by_id
-- Pull the resolved columns for the analysis population, keyed by the credit
-- case id so the caller can merge them onto whatever it already has.
-- Parameters: table, col_id, columns
SELECT
    {col_id},
    {columns}
FROM {table}
