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
- The one coherent cluster (market-breadth path stats `XS_BREADTH_MEAN20/STD20`, older-window
  momentum `SKIP_MOM_6_20/11_20`, `STOCH_20`, cross-sectional skew/kurtosis of `RET_1`,
  `XS_CORR_GPATH_G4`) block-tested at **+0.0050 ± 0.0023 on the chronology holdout** — but that
  holdout was used to select it. On views that played no part in selection it does not
  replicate: pseudo-test block +0.0022 (z 1.0, LightGBM) / +0.0012 (CatBoost); rolling blocks 1
  and 2 ≈ 0 for LightGBM and **−0.0040 (z −2.5) for CatBoost** on block 2. Verdict: selection
  bias, not a feature. Nothing from the screen is adopted.
- Whole families that are flat on both views: spectral/wavelet, entropy/fractal/scaling,
  polynomial shape, sign runs/streaks, autocorrelation/variance ratios, vol levels/ratios,
  oscillators (RSI/Bollinger), group lead/lag, per-date dispersion, volume path shape/trend,
  volume-return agreement, market-path beta/residuals, within-date kNN and "pairs" divergence,
  cross-sectional AR/regression coefficients. Fold-safe memory features are also flat or worse:
  kNN on 20-day / 5-day path shape (k = 50 / 200, historical next-day sign rate of the nearest
  training paths) −0.0009 to +0.0007; GMM-3 regime posteriors/state ≈ 0; sign-pattern / SAX
  hit-rates mixed-sign (one single-view hit among 14); fold-fit logistic momentum scores on the
  raw/normalised 20 returns −0.0030 to −0.0037 on the holdout (z −1.8 / −2.5). The
  cross-sectional theme as a whole has corr(cv_z, ho_z) = −0.01 across 126 candidates.
- Operational: don't run more than two screening processes at once on this machine (memory).

## Conventions
- Scratch/experiments outside the repo (session scratchpad or /tmp); only validated code in `src/`.
- Submissions in `submissions/` with a descriptive name; record leaderboard scores in
  `notes/accuracy_ceiling.md` and the README table.
- `git pull --rebase` before committing; small focused commits.
