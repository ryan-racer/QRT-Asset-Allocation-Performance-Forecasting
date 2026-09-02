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

**Recommendation:** `submissions/lgbm_final_nocat_mr_vol_fac_pinned.csv` (binary LightGBM,
base + rolling + mean-reversion + volume-regime + factor features, no allocation identity,
threshold pinned to the base rate; `src/qrt_replicate.py` spec `base,dum,miss,roll,mr,vol,fac`).
Expected leaderboard: ~0.516-0.521 given the ±0.006 block noise. `lgbm_binary_pinned.csv` is
the simpler fallback (same pipeline without mr/vol/fac).

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
