# Is the true date order recoverable?

**Status: CORRECTED (2026-09-02). Within-train adjacency IS recoverable by shape-matching; the
test set is clean, so this is a validation-purity issue, not an exploit.** The original
conclusion below (two clean nulls) was right about what it tested and wrong about what it
implied: the exact-match test in §2 fails because each row's returns are *rescaled* (so
`RET_1(t+1) == TARGET(t)` never holds bit-for-bit), not because adjacency was destroyed.
Matching the *direction* of the vectors instead — cosine similarity between a row's
`RET_2..RET_20` and another row's `RET_1..RET_19` for the same allocation — recovers
predecessors for ~54% of sampled rows at cosine > 0.99 (median best similarity 0.991), with many
date pairs matched unanimously across every allocation checked (e.g. `DATE_2167 → DATE_0001`,
`DATE_0002 → DATE_0370`). Public solutions report ~88% of train labels readable this way and
2,218/2,522 dates with a successor. The §3 autocorrelation test also stands: the *label
numbering* carries no order — the chains are a permutation of the labels, which is exactly why
label-order autocorrelation is null.

**Why it doesn't help the leaderboard:** the same check on `X_test` finds **0 of 1,374 sampled
test rows** with a predecessor anywhere in train or test — the organizers cleaned the test block.
Reconstructed chains can't reach test labels.

**Why it matters anyway:** adjacent train days share 19 of 20 return features. A plain
`TS`-grouped fold can put day *t* in train and its successor *t+1* in validation, so the model
sees a near-copy of the validation row's history — a mild optimism (public estimates ~0.001).
Chain-aware folds (assign whole chains to a fold) or an embargo remove it. Combined with the
much larger "dense-day" effect (see `notes/accuracy_ceiling.md`, validation section), this is
part of why random-date CV overstates leaderboard accuracy.

The original investigation follows, unchanged, for the record.

---

**Original status (2026-09-01): NOT RECOVERABLE — checked two independent ways, both clean nulls.**

The sibling QRT challenge (electricity price forecasting) had a `DAY_ID` column that was a
shuffle decoy while a second column, `ID`, silently encoded the true chronological order — worth
an exact-match dividend of +0.234 pooled Spearman once found (`../QRT-Electricity-Forecasting/notes/day_ordering.md`).
This challenge only ships one date-like column, `TS` (`DATE_0001` .. `DATE_2642`), so the natural
question is whether its numeric suffix is a similar decoy with real order recoverable underneath,
or whether the shuffle is real.

## 1. The one suspicious fact

`TS` numbers partition cleanly at the train/test boundary:

| split | `ts_num` range | n unique |
|---|---|---|
| train | 1 .. 2522 | 2522 |
| test | 2523 .. 2642 | 120 |

Zero overlap, zero gaps, and the ranges are each exactly as large as the unique-date count on
that side. That is not automatic — it means every integer in `1..2642` is used exactly once
across the whole dataset. This is the only structural hint that `ts_num` might carry real
ordering information (test appears to have been labelled *after* train), so it justified digging
further, the same way the coincidental `max(ID)-min(ID) == 1216` did in the electricity case.

It turned out to be a red herring for anything at the *row* level: it is consistent with train
and test dates simply being enumerated in two separate passes (train dates first, in whatever
order, then test dates), which says nothing about whether label order equals calendar order
*within* a split.

## 2. Exact-value fingerprint test

If `TS` order (or even just label-adjacency) reflected true daily adjacency, there is an exact
identity to exploit. Returns are literally defined as trailing windows: for allocation `S` at
real day `t+1`,

```
RET_1(t+1)  == TARGET(t)          # both are the realized return from t to t+1
RET_i(t+1)  == RET_{i-1}(t)       for i = 2..20
```

This is the same category of leak as the electricity challenge's `ID` — not a correlation, an
**exact equality** (up to float precision), giving a 20-way fingerprint per candidate adjacent
pair with an essentially zero false-positive rate if even a handful of the 19 secondary
identities hold.

**Method.** For every allocation, hash-join `TARGET` (rounded to 9 decimals) against `RET_1`
(rounded to 9 decimals) to find candidate pairs `(x, y)` where `RET_1(y) ≈ TARGET(x)`. Then check
how many of the 19 secondary identities (`RET_i(y) ≈ RET_{i-1}(x)`) also hold for each candidate.

**Results:**

| check | observed | null (permutation) | z |
|---|---|---|---|
| raw candidate count (`RET_1(y) ≈ TARGET(x)`) | 117 | 116.9 ± 0.30 (30 within-allocation shuffles of `target`) | **0.33** |
| secondary identities holding, loose tolerance (rtol=0.05) | max 3 of 19, distribution `{0:54, 1:46, 2:12, 3:5}` | random same-allocation row pairs: `{0:962, 1:706, 2:263, 3:54, 4:13, 5:1, 14:1}` | indistinguishable |

The raw candidate count is *exactly* what a random pairing of `target` and `RET_1` values
produces by chance (z = 0.33) — not the order-of-magnitude excess that would appear if a real
subset of adjacent pairs were mixed in with noise. And of the 117 "candidates" that did survive
the first join, **none** carried any of the 19 corroborating identities even loosely — their
match-count distribution is the same shape as a distribution built from literally random
same-allocation row pairs. There is no adjacent pair anywhere in the training set that survives
verification.

## 3. Autocorrelation / volatility-clustering test

The electricity challenge's cleanest validator was that real daily series show unmistakable
lag-1 autocorrelation and volatility clustering once placed in true order (`DE_NUCLEAR` 0.98,
`|GAS_RET|` 0.21, etc. — see the linked note), while a random ordering gives ~0. That test needs
no exact adjacency, so it is robust even if the labelled days are a sparse, irregular subsample
of the real calendar (in which case the fingerprint test in §2 could fail even under true
ordering). Built the day-level common factor (mean `target` and mean `|target|` across all
allocations present, indexed by `ts_num`) and checked its autocorrelation against a permutation
null (1000 shuffles of the day order, same series):

| lag | acf(mean target) | null mean ± std | z | acf(\|mean target\|) | z |
|---|---|---|---|---|---|
| 1 | -0.0179 | -0.0009 ± 0.0201 | -0.84 | -0.0021 | -0.14 |
| 2 | -0.0007 | -0.0004 ± 0.0205 | -0.01 | -0.0099 | -0.47 |
| 3 | -0.0106 | -0.0003 ± 0.0198 | -0.52 | 0.0277 | 1.42 |
| 5 | 0.0189 | -0.0015 ± 0.0199 | 1.03 | -0.0125 | -0.68 |
| 10 | 0.0498 | -0.0003 ± 0.0202 | 2.49 | -0.0195 | -0.97 |
| 20 | 0.0010 | -0.0003 ± 0.0205 | 0.07 | 0.0184 | 0.98 |

No lag clears significance under any reasonable multiple-comparison correction (largest is
z=2.49 at lag 10, alone among 12 tests, with no monotonic decay pattern — the signature of noise,
not of a lag-1-dominant real time series). Contrast with the electricity case's clean, monotonic,
uniformly-significant decay (z up to 36). There is no volatility clustering or serial dependence
along `ts_num` order.

## 4. Conclusion

Both tests — one an exact-match leak search, one a statistical time-series signature search —
come back clean null. Unless there is a transformation neither test would catch, `TS` order
(within a split) carries no recoverable calendar information. This is a meaningfully different
situation from the electricity challenge, and plausibly deliberate: the same provider ran both
challenges, and the electricity `ID` leak is exactly the kind of thing a second challenge would
be designed to close.

**Practical takeaway:** treat all training dates as exchangeable (as the challenge statement
says outright), don't build lag/order features off `TS`, and don't spend further effort trying to
reconstruct calendar time. `TS`-grouped cross-validation (no date split across train/val) is
already the correct, and only necessary, precaution — see [`../notebooks/eda.ipynb`](../notebooks/eda.ipynb).

## 5. Other angles checked, also negative

- **Allocation-number ordering**: `ALLOCATION_01`, `ALLOCATION_02`, ... does not block-sort by
  `GROUP` (blocks of consecutive allocation numbers sharing a `GROUP` average length 1.34, i.e.
  essentially randomly interleaved) — no hidden ordering there either.
- **Finer clustering within `GROUP`**: allocation-level turnover and `SIGNED_VOLUME_1`
  missingness rate show no significant between-`GROUP` separation (`turnover_median` F=0.006,
  p=0.999; `sv1_missing_rate` F=0.063, p=0.979) once aggregated to one row per allocation — the
  row-level `GROUP` differences seen in the EDA notebook are a composition effect (some groups
  have allocations that trade on more days), not evidence of finer within-group structure to
  recover. `ret1_std` shows a marginal difference (F=3.69, p=0.013) but nothing suggesting extra
  latent groups.
- **`sample_submission.csv`**: genuinely random (50.1/49.9 split, zero correlation with
  `sign(RET_1)` on the actual test features) — not a disguised leak of true test labels the way
  it was worth checking for in the electricity challenge.

## Reproducing

The checks above are one-off diagnostics, not a pipeline — re-run from a Python shell in the
project venv:

```python
import pandas as pd, numpy as np
X_train = pd.read_csv('data/raw/X_train.csv', index_col='ROW_ID')
y_train = pd.read_csv('data/raw/y_train.csv', index_col='ROW_ID')
df = X_train.join(y_train)
df['ts_num'] = df['TS'].str.extract(r'(\d+)').astype(int)
RET = [f'RET_{i}' for i in range(1, 21)]

# §2: fingerprint join
d = df.reset_index()
kf = d[['ALLOCATION', 'target']].assign(key=d['target'].round(9), idx=d.index)
kt = d[['ALLOCATION', 'RET_1'] + RET].assign(key=d['RET_1'].round(9), idx=d.index)
cand = kf.merge(kt, on=['ALLOCATION', 'key'], suffixes=('_x', '_y'))
cand = cand[cand['idx_x'] != cand['idx_y']]

# §3: day-level autocorrelation
day = df.groupby('ts_num').agg(mean_target=('target', 'mean')).sort_index()
acf1 = np.corrcoef(day.mean_target[:-1], day.mean_target[1:])[0, 1]
```
