# Asset Allocation Performance Forecasting

Trust or Short? Predicting the Performance of daily Asset Allocations

Provider: Qube Research & Technologies (QRT)

Started: Jan. 22, 2026

## Challenge Context

In the world of systematic trading, asset allocations are everywhere — but signal quality is
everything. Each day, traders are flooded with candidate allocations: portfolio constructions
based on recent signals, liquidity flows, or historical patterns. Some of these portfolios will
perform well in the next trading session. Others will underperform — or worse, underperform so
consistently that shorting them might be the more profitable move.

This challenge centers on a simple yet high-stakes question: can you predict whether a given
asset allocation is worth following — or shorting?

## What Is an Asset Allocation?

An asset allocation is a systematic method of constructing a portfolio of assets using
predefined signals or rules. In this challenge, each allocation is defined by a set of portfolio
weights, which can be positive or negative, applied on a specific day and held for one trading
session. From one day to another, an allocation can rebalance its weights to a certain
proportion, called turnover, depending on the rules used to compute it. The returns of an asset
allocation reflect the aggregated performance of these weighted positions rebalanced every day.

### Mathematical Definition

For a given trading day `t`, an allocation `S`, and `M` assets in a trading universe:

- The weights of allocation `S` at time `t`: `w_S,t = (w_S,t,1, w_S,t,2, ..., w_S,t,M)`
- The return of asset `i` from day `t` to day `t+1`: `r_i,t+1`

The realized return of allocation `S` at `t+1` is:

```
r_S,t+1 = sum_{i=1}^{M} w_S,t,i * r_i,t+1
```

## Challenge Goal

Each row in the dataset represents a day and an asset allocation, materialized as a portfolio
constructed and rebalanced on that day. The dataset gives a history of how that allocation
behaved when rebalanced over the past 20 trading days: its daily performance, liquidity
behaviour (proxied through weighted volumes), and median turnover.

The goal is to use that historical footprint to predict the sign of the allocation's performance
on the following day.

- If the model predicts positive return → trust the allocation.
- If the model predicts negative return → short the allocation.

## Evaluation Metric

Models are evaluated on accuracy: how often the model correctly predicts the direction (sign) of
an allocation's next-day return. Only the sign is evaluated — not the model's capacity to predict
the return's magnitude.

```
Accuracy = (1 / (T * M)) * sum_t sum_S 1[ sign(r_hat_S,t+1) == sign(r_S,t+1) ]
```

Where `T` is the number of timestamps, `M` is the number of allocations, and `1[.]` is the
indicator function.

## Data Description

The dataset is formatted as a time series with a multi-index of `(date, allocation)`. Each row
contains:

- 20-day history of allocation returns
- 20-day history of volume-weighted liquidity behavior
- Allocation median turnover
- Allocation anonymized `GROUP`
- Next-day allocation return (train only): the true performance for training; only its sign is
  evaluated on the test set.

### Columns

| Column | Description |
|---|---|
| `TS` | Timestamp of the snapshot. Dates are anonymized and shuffled — labels like `DATE_0001`, `DATE_0002` carry no guarantee of continuity. |
| `ALLOCATION` | Name of the allocation (`ALLOCATION_01` is the same allocation across `DATE_0001`, `DATE_0002`, etc.). |
| `RET_{i}` for `i` in 1..20 | Allocation's return on last day `i`. |
| `SIGNED_VOLUME_{i}` for `i` in 1..20 | Allocation's signed volume on last day `i` (see below). |
| `MEDIAN_DAILY_TURNOVER` | Allocation's median daily turnover (see below). |
| `GROUP` | Anonymized allocation group. |
| `TARGET` | Allocation's true next-day return. |

### Volumes, Weights, and Turnover

At every day `t`, each allocation `S` follows this property, given a universe of `M` trading
instruments:

```
for all t, for all S:  sum_{i=1}^{M} |w_S,i,t| = 1
```

The signed volume of allocation `S` at `t`:

```
V_S,t = sum_{i=1}^{M} w_S,t,i * V_i,t
```

where `V_i,t` is the total volume traded on the market for asset `i` during the trading session
at timestamp `t`. For homogeneity, these values are rescaled in a rolling fashion to ensure
comparability across different styles of allocations.

The median daily turnover of allocation `S` at `t`:

```
TO_S,t = sum_{i=1}^{M} |w_S,t,i - w_S,t-1,i|
MDT_S,t = median(TO_S,t, TO_S,t-1, ..., TO_S,t-20)
```

## Files

All files are indexed by a unique `ROW_ID`, referring to a unique tuple `(date, allocation)`,
allowing `X_train` to be mapped to `y_train`.

- `X_train.csv` — training set features (527,073 rows)
- `y_train.csv` — training set target
- `X_test.csv` — test set features (31,870 rows)
- `sample_submission.csv` — a random submission file in the correct format
- `benchmark_submission.ipynb` — a benchmark notebook that generates the leaderboard benchmark

## Benchmark Description

The benchmark notebook (`notebooks/benchmark_submission.ipynb`) builds:

- Additional features for the average historical performance of each allocation over multiple
  windows.
- Additional features for the average historical performance of all allocations over multiple
  windows.
- Additional features for the historical volatility of each allocation over the past 20 days.
- Additional features for the average historical volatility of all allocations over the past 20
  days.
- A Ridge regression fit on all base + additional features.
- A LightGBM model fit on all base + additional features, with a cross-validation section (this
  is the model submitted as the benchmark).

The public leaderboard score of the LightGBM benchmark is **0.5079** accuracy.

## Provider

Qube Research & Technologies (QRT) is a global quantitative and systematic investment manager,
operating in all asset classes across the world. Established in 2018, QRT has 2,000+ employees
across 16 offices globally. QRT supports coding initiatives and academic projects that promote
mathematics and science education.

## Contact

Questions about the challenge can be sent to `qrtdatachallenge@qube-rt.com`.
