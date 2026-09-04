# Project status (2026-09-03) — read this first if you're a second model/session

## Current best
- `submissions/catboost_nocat_mr_vol_fac_pinned.csv` — **0.5140 public leaderboard** (benchmark
  0.5079; ~0.52 is top quartile; private #1 is 0.5409). CatBoost (depth 6, lr 0.015, 300 it, no
  categorical features, 2 seeds) on the round-2 feature set, threshold pinned to the base rate.
- Pipeline: `src/qrt_replicate.py`, spec `base,dum,miss,roll,mr,vol,fac` (65 features, NO
  allocation identity). `notebooks/submission.ipynb` is the older round-1 model — superseded.

## The honest validation protocol (random-date CV overstates the leaderboard by ~0.01)
1. **Dense-day scoring**: test days have ~266 rows; score only train dates with >= 270 rows
   (`qrt_prep.dense_day_mask`). `qrt_prep.report(df, y, oof, threshold)` gives dense-day accuracy
   with day-clustered SE.
2. **Out-of-time holdout**: the train chronology is reconstructed (`notes/ts_chain_map.csv`);
   `qrt_prep.temporal_holdout(df, frac=0.15, embargo=20)` gives a leaderboard-like split
   (positive rate 0.498). Reference on its dense days: sign(RET_1) 0.5175, LightGBM round-2
   0.5231, CatBoost round-2 0.5273-0.5277 (pinned).
3. **Screening harness**: `src/qrt_screen.py` — `Screen().screen(candidates)` reports PAIRED
   day-clustered deltas on both views for each candidate on top of the round-2 base with a fast
   LightGBM (3 folds, 150 rounds). A candidate is only real if it helps on both views with the
   same sign; with hundreds of candidates, CV-only 2-SE wins are expected by chance.
4. Leaderboard noise is ±0.006-0.008 for a 120-day score. Don't read anything into smaller gaps.

## Ruled out (don't re-run; details in notes/accuracy_ceiling.md, notes/date_ordering.md)
- Any use of ALLOCATION identity (target encoding, hierarchical/slope/hit-rate encodings, hit-rate
  bucketing, native categoricals) — helps random-date CV, hurts out-of-time; the wide
  per-allocation base rates live in an early sparse era that the test block doesn't resemble.
- Per-TS cross-sectional ranks/z-scores, factor-projection tuning, per-GROUP models, GBM
  ensembles/blends, seed bagging, median-of-folds, dense-only or recency-weighted training,
  adversarial sample weights, GROUP-composition matching, near-duplicate cluster smoothing,
  day-level market-direction models, within-date demeaned targets, clipped-target MSE, Huber,
  "starved" LightGBM params, the 16-feature block from public repos, MLP/GRU sequence models,
  weekday (reporting-cycle) features even as an oracle.
- Chronology tricks on test: test days are isolated (no predecessor/successor anywhere).
- Label noise: a day's return is revised between target and next-day RET_1 (sign agreement
  88.5%) — a perfect model would score ~0.92, real models ~0.52-0.53.

## Feature screening (2026-09-03, in progress)
- Four themed screens through `src/qrt_screen.py` (ordinary technical, cross-sectional structure,
  volume/turnover/missingness, creative path-shape + fold-safe memory features): **214 unique
  candidates screened so far; 14 pass the lenient rule (same sign on both views, >= 1.5 SE on
  one) vs ~14 expected by chance.** Table: `notes/screen_results.csv` (cv/ho paired deltas, z).
- One coherent cluster is worth a confirmation pass (all <= 1.7 SE alone, consistently positive
  on both views): market-breadth path stats (`XS_BREADTH_MEAN20/STD20` = mean/std over the 20
  lag-days of the share of allocations positive), older-window momentum (`SKIP_MOM_6_20`,
  `SKIP_MOM_11_20`, `STOCH_20`), cross-sectional skew/kurtosis of `RET_1`. Pending: block test
  on seed 1 and the holdout, and in CatBoost.
- Whole families that are flat on both views: spectral/wavelet, entropy/fractal/scaling,
  polynomial shape, sign runs/streaks, autocorrelation/variance ratios, vol levels/ratios,
  oscillators (RSI/Bollinger), group lead/lag, per-date dispersion, volume path shape/trend,
  volume-return agreement. Still unscreened: the creative theme's fold-safe kNN path-similarity,
  sign-pattern hit-rates and GMM regime callables (running), and parts of the cross-sectional
  theme (market-path beta, within-date kNN).
- Operational: don't run more than two screening processes at once on this machine (memory).

## Conventions
- Scratch/experiments outside the repo (session scratchpad or /tmp); only validated code in `src/`.
- Submissions in `submissions/` with a descriptive name; record leaderboard scores in
  `notes/accuracy_ceiling.md` and the README table.
- `git pull --rebase` before committing; small focused commits.
