# Asset Allocation Performance Forecasting

This repository contains the QRT "Trust or Short?" asset allocation challenge materials, data,
and benchmark notebook. The task: given the last 20 trading days of an allocation's returns,
liquidity, and turnover, predict the sign of its next-day return.

## Project Structure

- `data/raw/`: original challenge CSV files
- `docs/`: challenge statement
- `notebooks/`: exploratory and benchmark notebooks
- `notes/`: written investigations (e.g. hidden-structure recovery attempts)
- `submissions/`: generated submission files

## Files

- `docs/challenge.md`: challenge statement, metric, and column reference
- `notebooks/eda.ipynb`: structure, descriptive statistics, target, missingness, distributions,
  outliers, feature/target association, train/test comparison, and grouped-CV baselines
- `notebooks/benchmark_submission.ipynb`: benchmark notebook, updated to use the local folder
  structure (reads from `../data/raw/`, writes to `../submissions/`)
- `notes/date_ordering.md`: investigation into whether the true calendar order is recoverable
  from `TS` (it isn't — see below)
- `data/raw/X_train.csv`: training input data (527,073 rows)
- `data/raw/y_train.csv`: training target data
- `data/raw/X_test.csv`: test input data (31,870 rows)
- `data/raw/sample_submission.csv`: random submission example in the correct format

## Running the Benchmark

Run `uv sync`, then open `notebooks/benchmark_submission.ipynb` and run the cells. It fits a
Ridge model and a cross-validated LightGBM model, writing both submissions to `submissions/`. The
LightGBM model is the one scored on the public leaderboard (0.5079 accuracy).

## EDA

`notebooks/eda.ipynb` — sanity checks, descriptive statistics, panel structure, target balance,
missingness, distributions, outliers, feature/target association, a train/test comparison, and
baselines under a `TS`-grouped CV harness (no date leaks across train/val, matching the
train/test split).

| baseline | grouped-CV accuracy |
|---|---|
| majority class | 0.5072 ± 0.0025 |
| sign(RET_1) | 0.5189 ± 0.0035 |
| ridge (41 features) | 0.5201 ± 0.0017 |
| lightgbm (41 features) | 0.5223 ± 0.0029 |

Main findings:

- **`RET_1` (yesterday's return) alone is most of the signal.** Its correlation with `TARGET` is
  ~0.085, far above any other lag or `SIGNED_VOLUME_i` (all near zero); `sign(RET_1)` alone
  already beats the published benchmark's 0.5079 public score. This is short-term continuation,
  not mean-reversion, and it holds within every `GROUP`.
- **`SIGNED_VOLUME_1` is missing 73.5% of the time, and it's structural, not random**: 24
  allocations (6 per `GROUP`) always report it, the rest are missing it ~80% of the time. Worth
  keeping as a missingness indicator rather than just imputing.
- **`GROUP` is a fixed attribute of the allocation** (never changes across its history) and the 4
  groups differ meaningfully in return volatility and typical turnover.
- **The `TS` numeric suffix carries no recoverable calendar order** — checked and ruled out the
  trick that worked on a sibling QRT challenge (`DAY_ID`/`ID`) two independent ways (an exact-match
  fingerprint search and an autocorrelation/volatility-clustering test against a permutation
  null); both come back clean nulls. Full writeup: [`notes/date_ordering.md`](notes/date_ordering.md).
- **Same-day targets across allocations correlate at ~0.26** — a shared daily factor exists. Not
  directly usable as a feature (it's built from the labels), but it's why cross-validation must
  be grouped by `TS`, and why the benchmark's same-day average-of-past-returns features are
  legitimate.
- **No data-quality issues**: zero duplicate rows or keys, no constant columns, no missing
  targets. Distributions are heavy-tailed but not corrupted — `SIGNED_VOLUME_i` has kurtosis in
  the hundreds, `MEDIAN_DAILY_TURNOVER` is right-skewed (≈4.3) with 13% of values beyond a
  3×IQR whisker, and `RET_1`'s extremes reach >40σ — genuine tail events, not glitches.
- **Train and test look like the same distribution**: matching means/spreads for `RET_1`,
  `SIGNED_VOLUME_1`, `MEDIAN_DAILY_TURNOVER`, and identical `GROUP` composition (same 278
  allocations on both sides); only `SIGNED_VOLUME_1`'s missing rate drifts slightly (73.5% train
  vs. 75.0% test). A `TS`-grouped CV score should transfer reasonably to the leaderboard.

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
