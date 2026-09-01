# How far can accuracy realistically go on this data?

**Status: extensively searched. Ceiling found at ~0.520-0.526 CV accuracy.**

The published benchmark scores 0.5079 on the public leaderboard. The EDA (`notebooks/eda.ipynb`)
found `sign(RET_1)` alone reaches 0.5189, and a first pass of feature engineering
(`notebooks/modeling.ipynb`) reached 0.5210-0.5223. This note documents a much larger search —
150+ configurations across five independent investigations — run to find out how much further
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

**Best validated result: CatBoost, binary objective, depth 6, learning rate 0.015, 300
iterations, with a fold-safe per-allocation target encoding (`k=50` shrinkage) on top of the
best feature set from `notebooks/modeling.ipynb` (`BASE_FEATURES + GROUP + SV1_MISSING +
rolling stats`). ~0.5257-0.5260 mean 5-fold `TS`-grouped CV accuracy, beating the prior best
in 9 of 10 fold-seed combinations tested (paired mean delta +0.0017, paired std 0.0010).**

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
