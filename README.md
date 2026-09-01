# Asset Allocation Performance Forecasting

This repository contains the QRT "Trust or Short?" asset allocation challenge materials, data,
and benchmark notebook. The task: given the last 20 trading days of an allocation's returns,
liquidity, and turnover, predict the sign of its next-day return.

## Project Structure

- `data/raw/`: original challenge CSV files
- `docs/`: challenge statement
- `notebooks/`: exploratory and benchmark notebooks
- `submissions/`: generated submission files

## Files

- `docs/challenge.md`: challenge statement, metric, and column reference
- `notebooks/benchmark_submission.ipynb`: benchmark notebook, updated to use the local folder
  structure (reads from `../data/raw/`, writes to `../submissions/`)
- `data/raw/X_train.csv`: training input data (527,073 rows)
- `data/raw/y_train.csv`: training target data
- `data/raw/X_test.csv`: test input data (31,870 rows)
- `data/raw/sample_submission.csv`: random submission example in the correct format

## Running the Benchmark

Run `uv sync`, then open `notebooks/benchmark_submission.ipynb` and run the cells. It fits a
Ridge model and a cross-validated LightGBM model, writing both submissions to `submissions/`. The
LightGBM model is the one scored on the public leaderboard (0.5079 accuracy).

## Data

Each row is a `(TS, ALLOCATION)` snapshot with:

- `RET_1..RET_20`: the allocation's daily returns over the last 20 trading days
- `SIGNED_VOLUME_1..SIGNED_VOLUME_20`: the allocation's rolling-rescaled signed volume over the
  same window
- `MEDIAN_DAILY_TURNOVER`: median of the allocation's daily turnover over the last 20 days
- `GROUP`: anonymized allocation group
- `TARGET` (train only): the allocation's next-day return — only its sign is scored

Dates are anonymized and shuffled, so `DATE_0001` → `DATE_0002` is not guaranteed to be
consecutive; allocation identity (`ALLOCATION_01`, etc.) is stable across dates. See
`docs/challenge.md` for the full metric and column definitions.
