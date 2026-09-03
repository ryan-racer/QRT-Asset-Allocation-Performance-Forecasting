"""Fast, honest feature-screening harness.

Screens candidate features one at a time (or as a block) on top of the round-2 base pipeline
with a small LightGBM, reporting PAIRED day-clustered deltas on two views:
  * dense-day TS-grouped CV (in-distribution), and
  * the chronology holdout (out-of-time, the leaderboard-like view).
A candidate only counts as real if it helps on both with a consistent sign; with hundreds of
candidates, CV-only wins at ~2 SE are expected by chance.

Usage:
    import qrt_screen as S
    sc = S.Screen()                       # loads data, caches per-fold base features + base OOF
    cand = pd.DataFrame({'ABS_RET1': sc.df['RET_1'].abs()}, index=sc.df.index)
    sc.screen(cand)                       # -> DataFrame: one row per candidate column
    sc.screen_block(cand)                 # -> one row for the whole block added at once
Fold-safe candidates that need the training fold (e.g. anything derived from target) are given
as callables: {'NAME': fn} with fn(train_df, other_df) -> (train_values, other_values).
"""
import os
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb

import qrt_prep as P
import qrt_replicate as R

BASE_SPEC = "base,dum,miss,roll,mr,vol,fac"
FAST_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
                   feature_fraction=0.8, lambda_l2=1.0, verbose=-1, seed=42, num_threads=4)
FAST_ROUNDS = 150
CACHE_DIR = os.environ.get("QRT_SCREEN_CACHE", "/tmp/qrt_screen_cache")


def paired_delta(hit_a, hit_b, ts):
    """Weighted per-day mean of (hit_b - hit_a) and its day-clustered SE."""
    return P.day_clustered_accuracy(np.asarray(hit_b, float) - np.asarray(hit_a, float), ts)


class Screen:
    def __init__(self, n_splits=3, seed=0, params=None, rounds=FAST_ROUNDS, holdout_frac=0.15,
                 embargo=20, data_dir="data/raw", verbose=True):
        self.params = dict(FAST_PARAMS, **(params or {}))
        self.rounds = rounds
        self.verbose = verbose
        X_train, y_train, self.X_test = P.load_raw(data_dir)
        self.df = X_train.join(y_train)
        self.y = (self.df["target"].to_numpy() > 0).astype(int)
        self.ts = self.df["TS"].to_numpy()
        self.dense = P.dense_day_mask(self.df)
        self.folds = P.make_folds(self.df["TS"], n_splits=n_splits, seed=seed)
        self.n_splits = n_splits
        self.ho_train, self.ho_hold = P.temporal_holdout(self.df, frac=holdout_frac, embargo=embargo)
        self.base_cols = R.feature_list(BASE_SPEC)
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._tag = hashlib.md5(f"{n_splits}-{seed}-{rounds}-{sorted(self.params.items())}-{holdout_frac}-{embargo}".encode()).hexdigest()[:10]
        self._build_base_frames()
        self._base_predictions()

    # ---- base features per split (fold-dependent parts computed from the training side only)
    def _build_base_frames(self):
        self.fold_frames = []
        for k in range(self.n_splits):
            va = self.folds == k
            tr_f, va_f = R.build_features(self.df[~va], self.df[va])
            self.fold_frames.append((R.to_frame(tr_f, self.base_cols).astype(np.float32),
                                     R.to_frame(va_f, self.base_cols).astype(np.float32)))
        tr_f, ho_f = R.build_features(self.df[self.ho_train], self.df[self.ho_hold])
        self.ho_frames = (R.to_frame(tr_f, self.base_cols).astype(np.float32),
                          R.to_frame(ho_f, self.base_cols).astype(np.float32))

    def _fit_predict(self, Xtr, ytr, Xva):
        ds = lgb.Dataset(Xtr, label=ytr)
        return lgb.train(self.params, ds, num_boost_round=self.rounds).predict(Xva)

    def _oof(self, extra=None):
        """extra: None, or list of (train_values_per_fold, val_values_per_fold) column arrays."""
        oof = np.zeros(len(self.df))
        for k in range(self.n_splits):
            va = self.folds == k
            Xtr, Xva = self.fold_frames[k]
            if extra is not None:
                Xtr = np.column_stack([Xtr] + [e[0][k] for e in extra])
                Xva = np.column_stack([Xva] + [e[1][k] for e in extra])
            oof[va] = self._fit_predict(Xtr, self.y[~va], Xva)
        return oof

    def _holdout(self, extra=None):
        Xtr, Xho = self.ho_frames
        if extra is not None:
            Xtr = np.column_stack([Xtr] + [e[0] for e in extra])
            Xho = np.column_stack([Xho] + [e[1] for e in extra])
        p = np.full(len(self.df), np.nan)
        p[self.ho_hold] = self._fit_predict(Xtr, self.y[self.ho_train], Xho)
        return p

    def _base_predictions(self):
        f = os.path.join(CACHE_DIR, f"base_{self._tag}.npz")
        if os.path.exists(f):
            z = np.load(f)
            self.base_oof, self.base_ho = z["oof"], z["ho"]
        else:
            self.base_oof, self.base_ho = self._oof(), self._holdout()
            np.savez(f, oof=self.base_oof, ho=self.base_ho)
        self.base_hit_cv = ((self.base_oof > 0.5).astype(int) == self.y)
        self.base_hit_ho = ((self.base_ho > 0.5).astype(int) == self.y)
        m = self.dense
        self.base_acc_cv = P.day_clustered_accuracy(self.base_hit_cv[m].astype(float), self.ts[m])
        h = self.ho_hold & self.dense
        self.base_acc_ho = P.day_clustered_accuracy(self.base_hit_ho[h].astype(float), self.ts[h])
        if self.verbose:
            print(f"base dense-day CV acc {self.base_acc_cv[0]:.4f} ± {self.base_acc_cv[1]:.4f} | "
                  f"holdout dense acc {self.base_acc_ho[0]:.4f} ± {self.base_acc_ho[1]:.4f}")

    # ---- candidate handling
    def _materialize(self, name, cand):
        """Returns ((per-fold train arrays, per-fold val arrays), (ho train array, ho array))."""
        if callable(cand):
            tr_list, va_list = [], []
            for k in range(self.n_splits):
                va = self.folds == k
                a, b = cand(self.df[~va], self.df[va])
                tr_list.append(np.asarray(a, np.float32).reshape(len(a), -1))
                va_list.append(np.asarray(b, np.float32).reshape(len(b), -1))
            a, b = cand(self.df[self.ho_train], self.df[self.ho_hold])
            ho = (np.asarray(a, np.float32).reshape(len(a), -1), np.asarray(b, np.float32).reshape(len(b), -1))
            return (tr_list, va_list), ho
        v = np.asarray(cand, np.float32)
        v = np.nan_to_num(v.reshape(len(v), -1))
        tr_list = [v[self.folds != k] for k in range(self.n_splits)]
        va_list = [v[self.folds == k] for k in range(self.n_splits)]
        return (tr_list, va_list), (v[self.ho_train], v[self.ho_hold])

    def _evaluate(self, name, extras):
        cv_extra = [e[0] for e in extras]
        ho_extra = [e[1] for e in extras]
        oof = self._oof(cv_extra)
        ho = self._holdout(ho_extra)
        hit_cv = ((oof > 0.5).astype(int) == self.y)
        hit_ho = ((ho > 0.5).astype(int) == self.y)
        m = self.dense
        d_cv, se_cv = paired_delta(self.base_hit_cv[m], hit_cv[m], self.ts[m])
        h = self.ho_hold & self.dense
        d_ho, se_ho = paired_delta(self.base_hit_ho[h], hit_ho[h], self.ts[h])
        row = dict(name=name, cv_delta=d_cv, cv_se=se_cv, cv_z=d_cv / se_cv if se_cv > 0 else np.nan,
                   ho_delta=d_ho, ho_se=se_ho, ho_z=d_ho / se_ho if se_ho > 0 else np.nan,
                   cv_acc=self.base_acc_cv[0] + d_cv, ho_acc=self.base_acc_ho[0] + d_ho,
                   pos_share_ho=float((ho[self.ho_hold] > 0.5).mean()))
        row["verdict"] = ("survivor" if (d_cv > 1.5 * se_cv and d_ho > 0) or (d_ho > 1.5 * se_ho and d_cv > 0)
                          else "noise")
        if self.verbose:
            print(f"{name:40s} cv {d_cv:+.4f} ± {se_cv:.4f} (z {row['cv_z']:+.1f}) | "
                  f"ho {d_ho:+.4f} ± {se_ho:.4f} (z {row['ho_z']:+.1f}) | {row['verdict']}", flush=True)
        return row

    def screen(self, candidates):
        """candidates: DataFrame (one column per candidate) and/or dict name -> array | callable."""
        items = []
        if isinstance(candidates, pd.DataFrame):
            items += [(c, candidates[c].to_numpy()) for c in candidates.columns]
        else:
            items += list(candidates.items())
        rows = []
        for name, cand in items:
            try:
                rows.append(self._evaluate(name, [self._materialize(name, cand)]))
            except Exception as e:  # keep screening the rest
                rows.append(dict(name=name, verdict=f"error: {e}"))
                if self.verbose:
                    print(f"{name}: ERROR {e}", flush=True)
        return pd.DataFrame(rows)

    def screen_block(self, candidates, name="block"):
        items = ([(c, candidates[c].to_numpy()) for c in candidates.columns]
                 if isinstance(candidates, pd.DataFrame) else list(candidates.items()))
        extras = [self._materialize(n, c) for n, c in items]
        return pd.DataFrame([self._evaluate(name, extras)])
