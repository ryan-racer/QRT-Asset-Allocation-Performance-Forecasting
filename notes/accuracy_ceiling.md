# How far can accuracy realistically go on this data?

**Status: extensively searched. Ceiling found at ~0.520-0.526 CV accuracy — but see the
validation section first: that CV number overstates the leaderboard by ~0.006-0.017.**

## Validation: why CV overstated the leaderboard (added 2026-09-02)

The CatBoost + allocation-encoding model below (0.5257 random-date CV) scored **0.5089** on the
public leaderboard — barely above the published benchmark (0.5079). Causes, each verified here:

1. **The test block is dense; half of train is sparse.** Test has 266 rows/day on average and
   102 of its 120 days have >= 270 allocations; train dates range from 19 to 276 rows and only
   1,273 of 2,522 are dense. Sparse days are easier (more one-directional). Scoring the same
   out-of-fold predictions only on dense train days:

   | model | all days | dense days (>= 270) | sparse days |
   |---|---|---|---|
   | sign(RET_1) | 0.5189 | **0.5132** | 0.5304 |
   | LightGBM, no encoding | 0.5208 | **0.5150** | 0.5325 |
   | LightGBM + allocation encoding | 0.5243 | **0.5185** | 0.5360 |

   Public solutions on this challenge report that dense-day CV lands within ~0.002 of their
   final leaderboard score. `src/qrt_prep.dense_day_mask` / `report()` implement this protocol
   with day-clustered standard errors (~±0.002 on 1,273 dense train days, ~±0.0065 on a
   120-day block).
2. **The allocation encoding doesn't transfer out-of-time.** It still adds +0.0035 in dense-day
   CV, but random-date folds can't see temporal decay; public analyses find per-allocation
   target bias has in-sample Spearman ~0.74 but only ~0.28 out-of-sample (day effects explain
   ~8% of target variance, allocation effects ~1.8%). Adversarial validation also shows several
   allocations present on only ~57% of train dates are present on ~95% of test dates, so the
   encoding was extrapolating for exactly the rows that dominate test. ALLOCATION as a native
   categorical is reported to generalize better (+0.0044 on the leaderboard for others).
3. **Leaderboard noise is large.** Same-day targets share a common factor (per-day positive rate
   ranges 0.15-0.90), so 120 test days behave like far fewer independent observations. Bootstrapping
   120-day blocks of our own OOF predictions gives SD 0.0069; a true-0.524 model scores below 0.5089
   only ~1.4% of the time, so the drop is real, but differences under ~0.006 on the leaderboard are
   noise. Public leaderboard scale: 0.5208 ≈ rank 245/1180 (top quartile); 0.5089 is bottom half.
4. **Positive-share drift.** MSE-regression models predict 57-59% positive vs a 50.7% base rate;
   accuracy is dominated by each day's majority sign, so a positive lean is costly on down days.
5. **Fold purity.** Adjacent train days are recoverable by shape-matching (see
   `notes/date_ordering.md`, corrected), so plain TS-grouped folds are mildly optimistic (~0.001).
6. **A true out-of-time holdout now exists.** The full train chronology was reconstructed
   (`notes/ts_chain_map.csv`; each `GROUP` trades on its own calendar, per-`GROUP` chains merged
   by topological sort, 2,520/2,522 dates ordered). `src/qrt_prep.temporal_holdout` takes the
   last 15% of the calendar (379 dates, 320 dense) with a 20-day embargo. Its positive rate is
   0.498 — balanced, like the leaderboard period (others' "always positive" probe scored 0.5033
   there). Verified numbers on the dense holdout days:

   | model | out-of-time acc ± SE | pos share |
   |---|---|---|
   | sign(RET_1) | 0.5175 ± 0.0039 | 0.505 |
   | round-2 pipeline (no allocation identity), thr 0.5 | **0.5231 ± 0.0041** | 0.593 |
   | same, threshold pinned to base rate | 0.5226 ± 0.0041 | 0.500 |
   | + per-allocation target encoding | 0.5191 | — |
   | + native allocation categoricals | 0.5199 | — |
   | recent-only training (25/50/75% of calendar) | 0.5176 / 0.5199 / 0.5218 | — |
   | time-decay weights (half-life 250-2000 days) | 0.5167-0.5242 | — |

   Allocation identity hurts out-of-time (the leaderboard result, reproduced); recency
   weighting does not beat training on everything. This holdout is now the primary selection
   criterion, ahead of dense-day CV and the adversarial pseudo-test block.

   **Rolling-origin confirmation** (three consecutive 378-date holdout blocks along the
   chronology, 20-day embargo, 2-seed average, pooled day-clustered SE ≈ 0.0024):

   | config | block 1 | block 2 | block 3 (latest) | mean | vs base |
   |---|---|---|---|---|---|
   | sign(RET_1) | 0.5124 | 0.5104 | 0.5168 | 0.5132 | −0.0036 |
   | round-2 pipeline (base) | 0.5140 | 0.5132 | 0.5231 | 0.5168 | — |
   | base, threshold → 51% / 55% positive | 0.5133 / 0.5145 | 0.5160 / 0.5154 | 0.5223 / 0.5225 | 0.5172 / 0.5175 | +0.0004 / +0.0007 |
   | + allocation target encoding | 0.5152 | 0.5119 | 0.5195 | 0.5155 | −0.0012 |
   | + native categoricals | 0.5126 | 0.5118 | 0.5207 | 0.5151 | −0.0017 |
   | dense-days-only training | 0.5089 | 0.5059 | 0.5197 | 0.5115 | −0.0053 |
   | recent 50% of dates only | 0.5122 | 0.5100 | 0.5202 | 0.5141 | −0.0026 |
   | time-decay weights, half-life 500 / 1000 | 0.5120 / 0.5139 | 0.5132 / 0.5141 | 0.5210 / 0.5242 | 0.5154 / 0.5174 | −0.0013 / +0.0007 |

   At test size (last 120 dates, 100 dense): base 0.5277 ± 0.0081, sign(RET_1) 0.5205 ± 0.0071,
   all-positive 0.5071 ± 0.0104 — **the SE of any 120-day score is ~0.008**, so the 0.5089
   leaderboard result sits ~1.5-2 SE below the honest out-of-time expectation of ~0.517-0.523.

   **Why the allocation encoding fooled CV.** The chronology is an expanding universe: ~505
   early dates with GROUP 3 only (65-69 rows/day), then ~500 dates with three groups, then
   ~1,500 dense dates with all four (the test block is dense). Per-allocation positive rates have
   within-era std **0.163 in the earliest era, 0.046 in the next, and 0.032-0.035 in the dense
   era** — barely above the ~0.023 expected from sampling noise. The "38.5%-65.9% base rates"
   that drove the +0.003 CV gain are an artefact of the tiny early universe; in the test-like era
   the persistent per-allocation component is worth at most +0.4 pt over all-positive.
   Correlation of allocation rates between consecutive eras falls 0.85 → 0.58 → 0.44 → 0.38.

   **Effects that are stable vs. not, along the timeline** (correlation with target > 0, ×1000,
   five consecutive blocks): RET_1 +85 / +71 / +28 / +30 / +35 (strong only in the sparse eras,
   ~+30 stable in the dense era); mean-reversion terms the same pattern; the common-factor part
   of RET_1 stable/rising (+14 → +42) while the idiosyncratic part decays (+25 → +12);
   SV1_MISSING / FULL_REPORT_DAY and turnover stable; SIGNED_VOLUME_1 and the per-date
   average-performance features **flip sign** across eras and are never significant.

   **Regime at the end of train / start of test**: the last 120 train dates have positive rate
   0.506 ± 0.010, trailing 20-day market direction ≈ 0 and below-average volatility; the test
   rows' own windows agree (share RET_1 > 0: 0.515 vs 0.511). The test period starts in a quiet,
   directionless regime, which is why an "always positive" probe scored only 0.5033 there.

The rest of this note is the original in-sample search, kept for the record; its numbers are
random-date CV and should be read ~0.006 lower for dense days.

## Round 2 (2026-09-02): what survives on test-like days

Two validation views are used below. **Dense-day CV**: 5-fold TS-grouped OOF scored only on
train dates with >= 270 allocations (`qrt_prep.report`, day-clustered SE ≈ ±0.002).
**Pseudo-test block**: the 250 dense train dates an adversarial train-vs-test classifier ranks
most test-like (69k rows; their true positive rate is 0.4997 vs 0.508 for the rest of train —
i.e. a more balanced period, like the leaderboard evidently was), trained on all other dates, SE
≈ ±0.004. The block is the closest thing to a temporal holdout this data allows.

**Replicated wins from public solutions, added cumulatively (dense-day CV, seed 0 / seed 1):**

| step | dense acc ± SE | pos share | paired Δ |
|---|---|---|---|
| MSE LightGBM, base features (prior baseline) | 0.5150 ± 0.0020 | 0.594 | — |
| binary logloss (lr 0.02, 15 leaves, 250 rounds) | 0.5191 ± 0.0019 | 0.596 | +0.0041 (half from params, half from the objective, z 2.4) |
| + ALLOCATION/GROUP as native categoricals | 0.5191 ± 0.0017 | 0.572 | +0.0000 |
| + per-TS cross-sectional relatives (42 cols) | 0.5186 ± 0.0018 | 0.581 | −0.0005 (dropped) |
| + mean-reversion `RET_1 − RET_MEAN_k` | 0.5199 ± 0.0018 | 0.591 | +0.0013 (z 1.8) |
| + volume-reporting regime flags | 0.5213 ± 0.0018 | 0.591 | +0.0015 (z 2.8) |
| + factor-projected RET_1 (common/idiosyncratic) | 0.5207 ± 0.0018 | 0.586 | −0.0007 |
| **final = cat + mr + vol + fac, no relatives** | **0.5215 ± 0.0019** / 0.5213 | 0.566 | +0.0065 vs baseline (z 4.6) |

Threshold pinning (predicted positive share forced to the 0.507 base rate) *hurts* in dense-day
CV (−0.001 to −0.002): the OOF probabilities are calibrated (bin (0.50, 0.505] → 51.3% up,
> 0.56 → 55.4% up), so the 57-59% lean is the conditional distribution on a mostly-up training
period, not miscalibration.

**The same pipeline on the pseudo-test block, where the period is ~50% positive:**

| model | thr 0.5 | pos | pinned to 0.507 | pos |
|---|---|---|---|---|
| sign(RET_1) | 0.5129 ± 0.0042 | 0.504 | — | — |
| binary base (interim `lgbm_binary_pinned`) | 0.5169 ± 0.0040 | 0.610 | 0.5178 ± 0.0039 | 0.496 |
| cat + mr + vol | 0.5173 ± 0.0039 | 0.594 | 0.5167 ± 0.0037 | 0.507 |
| final (cat + mr + vol + fac) | 0.5181 ± 0.0041 | 0.582 | 0.5200 ± 0.0041 | 0.501 |
| mr + vol only | 0.5196 ± 0.0044 | 0.617 | 0.5201 ± 0.0042 | 0.494 |
| **final minus cat (mr + vol + fac)** | 0.5202 ± 0.0046 | 0.622 | **0.5208 ± 0.0044** | 0.495 |
| earlier: LightGBM + allocation target encoding | 0.5156 | 0.580 | — | — |

Reading: on test-like days the allocation target encoding adds +0.0002 (vs +0.0035 in plain
dense-day CV) and native allocation categoricals are neutral-to-negative (−0.002) — allocation
identity does not transfer. Mean-reversion + volume-regime features do carry over (+0.003 over
the binary base). Pinning the threshold is neutral-to-positive here (+0.0005 to +0.002) even
though it hurts in CV, because this block — like the leaderboard period — is not up-leaning.
All of these differences are within ~1 SE of each other on 250 days; the ordering, not the
individual gaps, is what's trustworthy.

**Day-level market direction (negative).** Yesterday's cross-sectional mean return predicts the
date's positive rate (r = 0.108, t = 5.5; `sign(mean RET_1)` calls the day's majority 53.4% of
the time vs 51.6% always-up, strongest in high-vol and post-drawdown terciles). But it is not
exploitable: a perfect day-majority oracle would only reach 0.5716 row-level, the real day model
is unstable across folds (0.48-0.57), applying it to every row scores 0.506 (< sign(RET_1)), and
adding its prediction to the row model changes nothing (+0.0002). Reformulating the target as
within-date demeaned return hurts monotonically (−0.001 to −0.010). Row models already extract
what is there via RET_1.

## Round 3 (2026-09-02): construction forensics and the remaining public-solution ideas

**Where the ceiling comes from — the label is partly noise.** Along the reconstructed chronology
a day's return appears four times: as `target` on day t, then as `RET_1`, `RET_2`, `RET_3` on
the following rows. These versions are *revised* between rows: `target` vs the next row's
`RET_1` has sign agreement **88.5%** (corr 0.917; 72.5 / 94.1 / 98.9% by |target| tercile), and
the settled versions agree with `target` only 92.0-92.5%. The revision is portfolio-level and
structured (near-duplicate allocations share it, corr 0.95) but unpredictable from the row
(OOF R² 0.002). So ~8% of target signs are unknowable even given the settled return, and a
perfect model of the true return would score ~0.92, not 1.0 — the ~0.52 we reach is a small
edge on top of a noisy label, not a failure to find signal.

**Construction structure found, none of it usable on test:**

- SIGNED_VOLUME chains **bit-exactly** across consecutive rows (a rolling re-normalisation);
  RET is rescaled per row with a ratio ≈ 1 that carries no signal (corr with sign 0.0007).
- **Weekday cycle**: full-reporting days (SIGNED_VOLUME_1 present for all allocations) recur
  every 5 trading days. On dense days, 1-day momentum lives on two weekdays (`sign(RET_1)`
  accuracy 0.537 ± 0.005 on one, ~0.50 on two others; corr(RET_1, target) 0.150 vs ~0). The
  weekday is partially recoverable from a test date's cross-section (52% vs 26% chance) — but
  an **oracle** that hands the model the true weekday gains +0.0000 ± 0.0006 in CV and
  +0.0008 ± 0.0015 out-of-time (with RET_1 × weekday interactions +0.0018 ± 0.0016): the
  pipeline already captures it through `FULL_REPORT_DAY`. Lead closed.
- Near-duplicate allocations (91 pairs with return correlation > 0.9, one mirror pair at −0.95)
  have same-sign targets 82-93% of the time, but cluster-mean smoothing of predictions gains
  +0.0004 ± 0.0003 — the revision noise is shared within a cluster, so averaging can't remove it.
- Turnover is a slow allocation attribute (97.6% of variance is allocation identity; day-to-day
  corr 0.9997); its change carries nothing and is unavailable on test anyway.
- **Test days are isolated**: 0 exact-volume or cosine matches to any train or test row at any
  shift; every test day is ≥ 9 trading days from any other day — spaced samples from a period
  outside the contiguous train span. No leak reaches them.

**Remaining public-solution ideas, tested on all three views** (dense-day CV / pseudo-test block
/ out-of-time holdout, paired vs the round-2 pipeline): a 16-feature block from the top repos
(`reversal_1_vs_5`, `vol_ratio`, `turnover_per_volume`, EWMA/MACD, per-(TS, GROUP) positive rate
and Sharpe dispersion, …) is flat in CV and +0.002-0.003 on the two out-of-time views when the
threshold is pinned (≈1.5 SE, and ablating its top feature moves the views in opposite
directions — noise); per-allocation slope / momentum-hit encodings and hit-rate decile bucketing
are flat everywhere; a "starved" LightGBM (feature_fraction 0.08, min_data_in_leaf 2000, λ2 50)
loses 0.0014 in CV and gains ~0.002 out-of-time (≈1 SE); clipped-target MSE is a wash; Huber
regression is clearly worse (−0.0034 on the block, z −2.2); median-of-fold-models is noise;
CatBoost on the same features is +0.001 across views, as in round 1. Nothing clears ~2 SE.

## Round 4 (2026-09-03): a wide feature screen — 200+ candidates, nothing transfers

`src/qrt_screen.py` screens each candidate on top of the round-2 base with a fast LightGBM
(3 TS-grouped folds, 150 rounds) and reports paired day-clustered deltas on dense-day CV and on
the chronology holdout; a random-noise control reads as noise on both. Four themed screens
(ordinary technical; cross-sectional market structure; volume/turnover/missingness; creative
path-shape and fold-safe memory features) covered **214+ unique candidates**
(`notes/screen_results.csv`). **14 pass the lenient rule (same sign on both views, >= 1.5 SE on
one) — against ~14 expected by chance.** Whole families flat on both views: spectral/wavelet,
entropy/fractal/scaling, polynomial shape, sign runs/streaks, autocorrelation/variance ratios,
vol levels/ratios, oscillators, group lead/lag, per-date dispersion, market-path stats,
within-date kNN / "pairs" divergence, cross-sectional regressions, volume path shape/trend and
volume-return agreement.

The one coherent cluster — market-breadth path stats (mean/std over the 20 lag-days of the
share of allocations positive), older-window momentum (`mean(RET_6..20)`, `mean(RET_11..20)`,
path position in range), cross-sectional skew/kurtosis of `RET_1`, correlation with GROUP 4's
mean path — block-tests at +0.0050 ± 0.0023 (z 2.1) on the chronology holdout, but that holdout
was used to select it. On three views that played no part in selection it does not replicate
(full-strength models, paired deltas):

| view | LightGBM +cluster | CatBoost +cluster (xs part) |
|---|---|---|
| adversarial pseudo-test block (250 dates) | +0.0022 ± 0.0021 (z 1.0); all 11 feats +0.0036 (z 1.8) | +0.0012 ± 0.0016 |
| rolling block 1 (earlier chronology) | +0.0010 ± 0.0017 | −0.0002 ± 0.0012 |
| rolling block 2 | +0.0007 ± 0.0022 | **−0.0040 ± 0.0016 (z −2.5)** |

Selection bias, not signal. Nothing from the screen is adopted; the recommendation below stands.

Fold-safe "memory" features — the most novel candidates — are also flat or harmful on the
holdout: kNN on the standardized 20-day / 5-day path (historical next-day sign rate of the
50 / 200 nearest training paths, same-date rows excluded) −0.0009 to +0.0007; GMM-3 regime
posteriors ≈ 0; sign-pattern and SAX-trigram hit-rates mixed-sign (one single-view hit among
14 callables, the chance rate); a logistic "momentum score" fit on the raw or normalised 20
returns −0.0030 / −0.0037 (z −1.8 / −2.5). Across the cross-sectional theme's 126 candidates,
CV and holdout z-scores are uncorrelated (r = −0.01): the in-distribution and out-of-time
views agree only in that neither contains a usable new feature.

**Recommendation (updated after round 3):** `submissions/catboost_nocat_mr_vol_fac_pinned.csv`
— CatBoostClassifier (depth 6, learning rate 0.015, 300 iterations, seeds 42 and 7 averaged) on
the round-2 feature set (`src/qrt_replicate.py` spec `base,dum,miss,roll,mr,vol,fac`, no
allocation identity), threshold pinned to the base rate (predicts 50.2% positive on test). It is
the only round-3 change with a stable, >1.5 SE out-of-time gain: chronology holdout
0.5257-0.5267 at threshold 0.5 and **0.5273-0.5277 pinned** at both model seeds (LightGBM:
0.5227 / 0.5212), test-like block 0.5223 (LightGBM 0.5210), dense-day CV +0.0007-0.0012.
Expected leaderboard: ~0.518-0.523 given the ±0.008 SE at 120 days. **Actual public score
(2026-09-03): 0.5140** — up from 0.5089, +0.0076 over the public-period `sign(RET_1)` probe
(0.5064) versus +0.010 expected offline, i.e. ~0.5 SE below expectation and consistent with
the holdout. The public period is balanced (always-positive scores 0.5033) with roughly half
the training-era momentum edge, which caps what any model can score on it.
Caveat: the CatBoost gain is decisive only on the chronology holdout (+0.0054-0.0061 pinned,
z > 3 at both model seeds); on the adversarial test-like block it is +0.0005-0.0018, inside
that view's ±0.0017 SE. Blending LightGBM into CatBoost (0.25/0.75) and adding the 14
row-local public-repo features to CatBoost both cost 0.0004-0.0025 on the holdout, so the
plain CatBoost is kept. The earlier leaderboard disappointment (0.5089) was CatBoost *with* the
allocation encoding; the holdout attributes that loss to the encoding (−0.0036), not to the
learner. `lgbm_final_nocat_mr_vol_fac_pinned.csv` is the LightGBM version of the same pipeline
(91% agreement); `lgbm_binary_pinned.csv` the simpler fallback (no mr/vol/fac).

The published benchmark scores 0.5079 on the public leaderboard. The EDA (`notebooks/eda.ipynb`)
found `sign(RET_1)` alone reaches 0.5189, and a first pass of feature engineering
(`notebooks/modeling.ipynb`) reached 0.5210-0.5223. This note documents a much larger search —
150+ configurations across eight independent investigations — run to find out how much further
that number can legitimately move, and to have a clear, evidenced answer for "why not higher"
before spending more effort chasing it.

## Summary table

| investigation | scope | best found | beats prior best? |
|---|---|---|---|
| Feature engineering | ~20 new features (cross-sectional rank/z-score, momentum streaks, vol-normalized returns, volume/turnover interactions, refined target encodings), individually and combined, against an established noise floor | 0.5242 ± 0.0002 (ties baseline) | No — net zero after accounting for noise |
| Per-allocation target encoding | Fold-safe shrunk mean of `target` per `ALLOCATION` (see below) | **0.5243** (LightGBM) | **Yes — the single biggest lever found**, +0.0033 over the 0.5210 pre-encoding baseline |
| GBM hyperparameter/model search | ~100 configs across LightGBM, XGBoost, CatBoost; regression (`mse`) vs. binary-classification objectives; depth/leaves/learning-rate/regularization sweeps | 0.5257-0.5260 (CatBoost, binary, depth 6) | Yes — modest, +0.0014-0.002 over the flat-encoding LightGBM baseline |
| Hierarchical (partial-pooling) allocation encoding | 21-config grid: shrink allocation → `GROUP` mean → global mean, vs. flat shrink straight to global mean | 0.5253 (ties flat encoding, never exceeds it) | No |
| Ensembling | Naive blend, rank-transformed blend, nested-CV-weighted blend of LightGBM + CatBoost + Ridge | 0.5257 (nested-weighted, ties best single model) | No — no diversity dividend beyond the single best model |
| Neural sequence models | MLP and GRU (treating `RET_1..20`/`SIGNED_VOLUME_1..20` as an ordered sequence, not 40 independent columns), with a learned per-allocation embedding implemented but not fully evaluated | 0.5237 (MLP pilot) | No — underperforms GBMs |
| CatBoost native categoricals | `ALLOCATION`/`GROUP` passed directly as CatBoost categorical features (ordered target statistics, a different leak-safe algorithm than the manual encoding), alone and combined with the manual encoding + cross-sectional features | 0.5259 alone, 0.5258 combined | No — ties the manual-encoding result; combined version has the tightest fold-to-fold spread of anything tried |
| 5-seed bagging | Averaging CatBoost-winner predictions across 5 random training seeds, to check whether the model's own training randomness was hiding signal | 0.5256 (vs. 0.5257 single-seed) | No — identical within noise |

**Best validated result: CatBoost, binary objective, depth 6, learning rate 0.015, 300
iterations, with a fold-safe per-allocation target encoding (`k=50` shrinkage) *and*
`ALLOCATION`/`GROUP` as native categoricals, on top of the best feature set from
`notebooks/modeling.ipynb` (`BASE_FEATURES + GROUP + SV1_MISSING + rolling stats +
cross-sectional`). ~0.526 mean 5-fold `TS`-grouped CV accuracy — this combined config is what
`notebooks/submission.ipynb` trains; it doesn't beat the plain manual-encoding version on mean,
but has the tightest fold-to-fold spread of anything tried in this search.**

## The one real lever: per-allocation target encoding

Per-allocation base rates range **38.5% to 65.9% positive**, with **0.67 split-half
reliability** — a real, persistent effect (some allocations are structurally more often "up"
than others), far stronger than anything `GROUP` captures on its own (0.500-0.510 range). This
must be computed fold-safe (from training-fold rows only) since it's derived from `target`:

```python
def add_alloc_encoding(train_df, val_df, k=50):
    global_mean = train_df['target'].mean()
    stats = train_df.groupby('ALLOCATION')['target'].agg(['mean', 'count'])
    shrunk = (stats['count'] * stats['mean'] + k * global_mean) / (stats['count'] + k)
    return (train_df['ALLOCATION'].map(shrunk).fillna(global_mean).to_numpy(),
            val_df['ALLOCATION'].map(shrunk).fillna(global_mean).to_numpy())
```

This alone took LightGBM from 0.5210 to 0.5243 — bigger than every other feature tried,
combined. `src/qrt_features.py` has this as `add_alloc_encoding`.

## Why nothing else moved it further

**Feature space is exhausted for this model shape.** ~20 additional engineered features (rank
within `TS`, z-scores, momentum/streak counts, RET-volume interactions, turnover interactions,
vol-normalized returns, a per-(allocation, RET_1-sign) refined encoding) were tested
individually and combined, against a properly-established noise floor (the *same* feature set
re-run across 3 fold-seeds gave mean 0.5242, std 0.0002, range 0.0006 — so anything smaller than
~0.0003-0.0006 isn't distinguishable from fold-assignment noise). None survived. A "kitchen
sink" of the best-looking individual features, combined and re-tested at full scale across 3
seeds, nets to -0.0001 — worse than doing nothing. The one statistically clear result was
negative: further conditioning the allocation encoding on `RET_1`'s sign fragments the sample
and makes it reliably worse (-0.0014).

**Model capacity/objective gave a real but small edge, then plateaued.** Switching from MSE
regression (predict `target`, threshold the sign) to a binary classification objective
(directly predict `sign(target)`) helped consistently by +0.0002-0.0006 across all three GBM
libraries — makes sense, since accuracy (not MSE) is the actual metric. CatBoost's ordered
boosting handled deeper trees (depth 5-6) better than LightGBM/XGBoost (which peaked shallow,
depth 3-4). But extensive refinement around the winning region (dozens of nearby
learning-rate/depth/iteration combinations) never found anything past ~0.526.

**Hierarchical shrinkage doesn't help because `GROUP` is too weak a prior.** Shrinking each
allocation toward its `GROUP`'s mean (instead of straight to the global mean) was tested across
a 21-point k1×k2 grid. It never beat flat shrinkage — makes sense given `GROUP`-level base rates
only span 0.500-0.510, so `GROUP` doesn't cluster allocation base rates tightly enough to be a
better shrinkage anchor than the raw global mean.

**Ensembling has no diversity dividend.** LightGBM, CatBoost, and Ridge OOF predictions were
rank-transformed and blended three ways: naive equal-weight (worse — a weak Ridge component
drags the average down, the same failure mode as an earlier 4-model naive blend that scored
0.5228 against a 0.5243 single-model baseline), grid-searched weights (also worse), and a
properly nested-CV logistic-regression weighting (ties the best single model, +0.00004 — noise).
The models aren't different enough in what they get right for blending to add anything.

**Neural sequence models underperform, not overfit.** An MLP and a GRU (treating `RET_1..20`/
`SIGNED_VOLUME_1..20` as an actual ordered sequence, which GBMs can't do — they see 40
independent scalar columns) were tuned and evaluated on the identical folds. Both landed at or
below the GBM band (MLP 0.5237, GRU 0.5226 on completed folds) with no sign of a hidden edge —
not a leakage red flag (no suspiciously large win to chase), just a genuinely weak, close-to-
linear signal that a plain gradient-boosted tree already captures about as well as a neural net
can from this little data per allocation.

**CatBoost's native categorical handling doesn't beat the manual encoding, and averaging away
training randomness doesn't reveal hidden signal either.** As one final check (after everything
above independently converged on the same ~0.525 band), `ALLOCATION`/`GROUP` were passed directly
to CatBoost as categorical features (`cat_features=[...]`) instead of the manual shrunk-mean
encoding — CatBoost computes its own ordered target statistics internally, entirely from the
training-fold rows passed to `.fit()`, a different algorithm that could plausibly have picked up
something the manual encoding's flat shrinkage missed. It landed at 0.5259 alone and 0.5258
combined with the manual encoding and cross-sectional features — indistinguishable from the
0.5257 manual-encoding-only result. Separately, averaging CatBoost predictions across 5 different
random training seeds (bagging away the model's own training-randomness variance) gave 0.5256 —
also identical within noise. Eight independent methods, same ceiling.

## Why this ceiling is expected, not a failure of effort

Published research on short-horizon direction prediction (drawn from short-term reversal/
momentum literature, volume-return studies, and quant-competition postmortems) puts single-stock
daily-direction accuracy in the 50-53% range out-of-sample, with
claims materially above 55-57% usually attributable to leakage or overfitting rather than real
edge. Numerai — the closest live analog to this problem shape (cross-sectional panel, scored on
correlation) — treats ~0.02-0.03 correlation as a *good* score, which translates to only a
couple of accuracy points above base rate. The gain found here (0.5079 published benchmark →
0.526 best validated result, ≈1.8 points) sits squarely inside that "genuine edge from careful
feature/model work" range the literature predicts. Reaching materially higher (e.g. 0.55) would
require either a source of signal not present in this 41-column feature set, or would not
survive honest out-of-sample validation.

## Reproducing

Minimal version (manual encoding only, ~0.5257); the exact adopted pipeline (manual encoding +
native categoricals + cross-sectional features, ~0.526 with tighter fold spread) is
`notebooks/submission.ipynb`.

```python
import sys; sys.path.insert(0, 'src')
import numpy as np
import qrt_prep as P, qrt_features as F
from catboost import CatBoostClassifier

X_train, y_train, X_test = P.load_raw('data/raw')
df = F.engineer(X_train.join(y_train))
y = df['target'].to_numpy()
y_sign = (y > 0).astype(int)
folds = P.make_folds(df['TS'], n_splits=5, seed=0)
BASE = P.BASE_FEATURES + F.GROUP_DUMMY_COLS + F.MISSING_COLS + F.ROLLING_COLS

accs = []
for fold in range(5):
    val_mask = folds == fold
    tr, va = df[~val_mask], df[val_mask]
    tr_enc, va_enc = F.add_alloc_encoding(tr, va, k=50)
    Xtr = np.column_stack([F.to_matrix(tr, BASE), tr_enc])
    Xva = np.column_stack([F.to_matrix(va, BASE), va_enc])
    model = CatBoostClassifier(iterations=300, learning_rate=0.015, depth=6,
                                random_seed=42, verbose=False, thread_count=8)
    model.fit(Xtr, y_sign[~val_mask])
    pred = model.predict_proba(Xva)[:, 1]
    accs.append(((pred > 0.5).astype(int) == y_sign[val_mask]).mean())
print(np.mean(accs), accs)  # ~0.526
```
