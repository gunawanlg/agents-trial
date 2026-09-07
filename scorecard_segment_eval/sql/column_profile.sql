-- query: column_profile
-- Value distribution of one candidate column, so its cardinality can be
-- checked before it is adopted as a segmentation column.
-- Parameters: table, column, col_id, sample_limit
SELECT
    {column}              AS value,
    COUNT(*)              AS n_rows,
    COUNT(DISTINCT {col_id}) AS n_cases
FROM (
    SELECT {col_id}, {column}
    FROM {table}
    LIMIT {sample_limit}
) sampled
GROUP BY {column}
ORDER BY n_rows DESC
