-- 1. Create labeled dataset with target variable and 3-day rolling sentiment feature
CREATE OR REPLACE TABLE stock_data.sentiment_labeled AS
SELECT
  *,
  IF(next_day_pct_change > 0, 1, 0) AS moved_up,
  AVG(mean_sentiment) OVER (
    PARTITION BY company
    ORDER BY date
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS sentiment_3day_avg
FROM
  stock_data.sentiment_returns
WHERE
  next_day_pct_change IS NOT NULL;

-- 2. Train Logistic Regression model to predict next-day movement
CREATE OR REPLACE MODEL stock_data.movement_predictor
OPTIONS(
  model_type='logistic_reg',
  input_label_cols=['moved_up']
) AS
SELECT
  mean_sentiment,
  pct_change,
  headline_count,
  sentiment_3day_avg,
  moved_up
FROM
  stock_data.sentiment_labeled;

-- 3. Evaluate Logistic Regression model performance
SELECT * FROM ML.EVALUATE(MODEL stock_data.movement_predictor);

-- 4. Alternative: Boosted Tree Classifier (commented out)
-- CREATE OR REPLACE MODEL stock_data.movement_predictor_boosted
-- OPTIONS(
--   model_type='boosted_tree_classifier',
--   input_label_cols=['moved_up']
-- ) AS
-- SELECT
--   mean_sentiment,
--   pct_change,
--   headline_count,
--   sentiment_3day_avg,
--   moved_up
-- FROM
--   stock_data.sentiment_labeled;
