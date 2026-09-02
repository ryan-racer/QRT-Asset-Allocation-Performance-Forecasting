# Asset Allocation Performance Forecasting

This repository contains the QRT "Trust or Short?" asset allocation challenge materials, data,
and benchmark notebook. The task: given the last 20 trading days of an allocation's returns,
liquidity, and turnover, predict the sign of its next-day return.

## Project Structure

- `data/raw/`: original challenge CSV files
- `docs/`: challenge statement
- `notebooks/`: exploratory and benchmark notebooks
- `notes/`: written investigations (e.g. hidden-structure recovery attempts)
- `src/`: shared data loading, feature engineering, and the CV harness
- `submissions/`: generated submission files

## Files

- `docs/challenge.md`: challenge statement, metric, and column reference
- `notebooks/eda.ipynb`: structure, descriptive statistics, target, missingness, distributions,
  outliers, feature/target association, train/test comparison, and grouped-CV baselines
- `notebooks/benchmark_submission.ipynb`: benchmark notebook, updated to use the local folder
  structure (reads from `../data/raw/`, writes to `../submissions/`)
- `notes/date_ordering.md`: investigation into whether the true calendar order is recoverable
  from `TS` (it isn't — see below)
- `notes/accuracy_ceiling.md`: a 150+ configuration search for how far accuracy can realistically
  go on this data (feature engineering, 3 GBM families, hierarchical encoding, ensembling, neural
  sequence models) — see "Extended search" below
- `notebooks/modeling.ipynb`: turns the EDA findings into features and checks each one under CV
- `notebooks/submission.ipynb`: trains the round-1 CatBoost model on the full training set and
  writes `submissions/catboost_alloc_encoding.csv` (superseded — scored 0.5089 on the leaderboard)
- `src/qrt_prep.py`: raw loading, feature-column constants, `TS`-grouped fold/CV harness, and the
  honest validation helpers (`dense_day_mask`, `day_clustered_accuracy`, `oof_predictions`, `report`)
- `src/qrt_features.py`: `GROUP` dummies, `SIGNED_VOLUME_1` missingness indicator, rolling return
  stats, same-day cross-sectional features, fold-safe per-`ALLOCATION` target encoding
- `src/qrt_replicate.py`: the round-2 pipeline — binary-objective LightGBM with mean-reversion,
  volume-reporting-regime and factor-projected `RET_1` features (`build_features`,
  `make_fit_predict`, `FEATURE_SETS`)
- `submissions/lgbm_final_nocat_mr_vol_fac_pinned.csv`: **current recommended submission**
  (round-2 pipeline without allocation identity, threshold pinned to the base rate);
  `lgbm_binary_pinned.csv` is the simpler fallback; `diag_*.csv` are leaderboard calibration probes
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

## Feature engineering

`notebooks/modeling.ipynb` turns the EDA findings into features (`src/qrt_features.py`) and
checks each one under the same `TS`-grouped CV harness (`src/qrt_prep.py`), added cumulatively
on top of the benchmark's own 41-feature set, averaged over 2 fold-seeds (10 fold estimates per
number):

| feature set | ridge | lightgbm |
|---|---|---|
| baseline (benchmark features) | 0.5194 | 0.5210 |
| + `GROUP` dummies | 0.5194 | 0.5210 |
| + `SIGNED_VOLUME_1` missingness indicator | 0.5197 | 0.5211 |
| + rolling return stats (`RET_MEAN_5/20`, `RET_STD_20`) | 0.5198 | **0.5212** |
| + cross-sectional same-day features | 0.5206 | 0.5212 |
| + `RET_1 x GROUP` interaction | **0.5207** | 0.5211 |

Real, but small: cumulative gains top out around +0.0012 accuracy (ridge, cross-sectional
features) and +0.0002–0.0004 (lightgbm, mostly rolling stats). `GROUP` alone is worth essentially
nothing as a raw feature (+0.00006 lightgbm), so it was tried as an explicit `RET_1 x GROUP`
interaction on top of everything else too — also close to nothing (+0.00004 ridge over
cross-sectional alone, and it slightly *hurts* lightgbm). The cross-sectional same-day features
already seem to capture whatever `GROUP`-conditional signal is there; an explicit interaction
term doesn't add more. Every configuration here already clears the published benchmark's 0.5079
public score by 1–1.4 points. LightGBM feature importance (gain) on the best set confirms the EDA
directly in a trained model: `RET_1` is ~10x the next feature (`RET_MEAN_5`), `GROUP` dummies and
the missingness indicator don't crack the top 15.

## Extended search: how far can accuracy go?

`notes/accuracy_ceiling.md` documents a 150+ configuration search — exhaustive feature
engineering against an established noise floor, LightGBM/XGBoost/CatBoost hyperparameter tuning
across regression and classification objectives, hierarchical (partial-pooling) allocation
encoding, properly nested-CV-weighted ensembling, and MLP/GRU neural sequence models — run to see
how much further CV accuracy could legitimately move past the feature-engineering table above.

The one real additional lever: a **fold-safe per-`ALLOCATION` target encoding**
(`qrt_features.add_alloc_encoding`). Per-allocation base rates range **38.5%-65.9% positive**
with **0.67 split-half reliability** — far more reliable than `GROUP`'s narrow 0.500-0.510 range
— so it must be computed inside each CV fold (it uses `target`) but is otherwise the single
biggest lever in the whole project: LightGBM 0.5210 → 0.5243. On top of that, switching to
**CatBoost with a binary-classification objective** (depth 6, learning rate 0.015) adds another
step to **~0.526** mean CV accuracy.

Everything else tried — ~20 more engineered features, hierarchical shrinkage toward `GROUP`
instead of the global mean, ensembling 3 model families, and neural sequence models over the raw
`RET_1..20`/`SIGNED_VOLUME_1..20` history — failed to beat that ~0.526 ceiling; full results and
reasoning in `notes/accuracy_ceiling.md`. This lines up with published research on short-horizon
direction prediction (single-stock daily-direction accuracy is typically reported at 50-53%
out-of-sample; claims materially above 55-57% are usually leakage or overfitting artifacts) — the
gain found here (0.5079 published benchmark → 0.526, ≈1.8 points) sits squarely inside the range
that literature predicts as a genuine, defensible edge for this problem shape.

## Leaderboard reality check and round 2

The CatBoost + allocation-encoding model above scored **0.5089** on the public leaderboard —
1.7 points below its CV and barely above the benchmark. The validation was the bug, not the
model. Verified causes and fixes (full detail in `notes/accuracy_ceiling.md`, validation and
round-2 sections):

- **Score only dense days.** Test has 266 rows/day; half of train's dates are sparse, and sparse
  days are easier. Dense-day CV (`qrt_prep.report`) drops every model ~0.006 and matches what
  public solutions see on the leaderboard (~0.52 is top quartile).
- **Allocation identity doesn't transfer.** On the 250 most test-like train dates (found by
  adversarial validation; their positive rate is 50.0%, like the leaderboard period evidently
  was) the target encoding adds +0.0002 and native categoricals are neutral-to-negative.
- **Positive lean.** Regression/binary models predict 57-62% positive; on a balanced period that
  costs. Pinning the threshold to the base rate is neutral in CV and slightly positive on the
  test-like block.
- **What does carry over** (replicated from public solutions, `src/qrt_replicate.py`): binary
  logloss (+0.004), mean-reversion terms `RET_1 − RET_MEAN_k` (+0.001) and volume-reporting
  regime flags (+0.0015). Cross-sectional rank/z features, factor projection, per-`GROUP` models,
  ensembling, and a day-level market-direction model were all tested and add nothing.
- **Leaderboard noise is ±0.006** on 120 days (same-day common factor), so differences below
  that between submissions are not interpretable.

| model | dense-day CV | pseudo-test block |
|---|---|---|
| sign(RET_1) | 0.5132 | 0.5129 |
| binary LightGBM, base features, pinned (`lgbm_binary_pinned.csv`) | 0.5177 | 0.5178 |
| **round-2 pipeline, no allocation identity, pinned (recommended)** | ~0.520 | **0.5208 ± 0.0044** |
| round-1 CatBoost + allocation encoding (`catboost_alloc_encoding.csv`) | 0.5185 (LightGBM equiv.) | 0.5156 → **0.5089 real** |

## Submission (round 1, superseded)

`notebooks/submission.ipynb` trains the round-1 model — CatBoost, binary objective,
depth 6, learning rate 0.015, with the fold-safe allocation encoding — on the full training set
and writes `submissions/catboost_alloc_encoding.csv` (~0.526 CV accuracy, re-verified in the
notebook itself), format-checked against `sample_submission.csv` (same shape, same index, values
in `{0, 1}`).

A now-superseded LightGBM submission (`submissions/lightgbm_rolling_stats.csv`, ~0.521 CV
accuracy) is left in place from an earlier iteration.

One check worth flagging on the earlier LightGBM model: its raw out-of-fold predictions on train
were skewed positive (59% > 0, vs. the true 50.7% base rate), which looked like it might call for
recentering the `sign(pred) > 0` decision rule. Tested with a double cross-validated threshold
search (pick the accuracy-best cutoff from other folds' OOF predictions, score the held-out fold
with it, so the threshold never sees the labels it's evaluated against) — tuning makes no
difference (0.5208 vs. 0.5207 threshold=0). Accuracy depends on where the sign genuinely flips
relative to truth, not on matching the marginal predicted-positive rate, so `sign(pred) > 0` (or
`predict_proba > 0.5` for CatBoost) is the right rule as-is.

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
