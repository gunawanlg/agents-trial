-- query: fetch_portfolio
-- Portfolio mix of the credit cases in the analysis table.  The dominant
-- portfolio drives the target / observation-flag auto-detection.
-- Parameters: table, col_id, col_portfolio, sample_limit
SELECT
    {col_portfolio}          AS portfolio,
    COUNT(DISTINCT {col_id}) AS n_cases
FROM (
    SELECT {col_id}, {col_portfolio}
    FROM {table}
    LIMIT {sample_limit}
) sampled
GROUP BY {col_portfolio}
ORDER BY n_cases DESC
