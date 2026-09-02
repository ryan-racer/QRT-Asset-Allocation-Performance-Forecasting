"""Replication of the techniques other teams reported as leaderboard gains on this challenge,
evaluated under the honest protocol (TS-grouped 5-fold OOF, dense-day accuracy, day-clustered SE).
Drop-in for src/: only depends on qrt_prep.

    from qrt_replicate import build_features, fit_predict, FEATURE_SETS
    oof = qrt_prep.oof_predictions(df, y, fit_predict, n_splits=5, seed=0)   # y = raw target
    qrt_prep.report(df, y, oof, threshold=0.5)                              # oof are probabilities

Feature layers, by what they are allowed to see:
  add_local(df)                 row-local transforms
  add_per_date(df)              per-TS and per-(TS, GROUP) aggregates. Identical whether computed on
                                a fold, on all of train, or on X_test alone (folds are TS-grouped and
                                test is scored in whole-date batches), so no leakage channel.
  add_fold_dependent(tr, other) fit on the TRAINING rows' *features* only (never the target):
                                per-allocation SIGNED_VOLUME_1 reporting rate and the RET_1 factor
                                model (top-k eigenvectors of the train-fold allocation correlation
                                matrix; each date is then ridge-projected using only its own RET_1).
build_features(train_df, other_df) = all three; returns (train_feats, other_feats).

Dense-day OOF accuracy (seed 0 / seed 1), LightGBM binary, 250 rounds, lr 0.02, 15 leaves:
  FEATURE_SETS['final']  (base + GROUP dummies + SV1_MISSING + rolling + native categoricals
                          + mean-reversion + volume-regime + factor projection)  0.5215 / 0.5200
  FEATURE_SETS['s5']     (same but cross-sectional relatives instead of factor)  0.5213 / 0.5204
  MSE baseline (depth 3, lr 0.01)                                                0.5150 / ...
"""
import numpy as np
import pandas as pd

from qrt_prep import BASE_FEATURES, RET

XS_BASE = ["RET_1", "RET_MEAN_5", "RET_MEAN_20", "RET_STD_20", "MEDIAN_DAILY_TURNOVER"]
N_FACTORS = 8
CAT_COLS = ["ALLOC_CODE", "GROUP_CODE"]  # native categoricals (NOT a target encoding)

GROUPS = {
    "base": BASE_FEATURES,
    "dum": [f"GROUP_{g}" for g in [1, 2, 3, 4]],
    "miss": ["SV1_MISSING"],
    "roll": ["RET_MEAN_5", "RET_MEAN_20", "RET_STD_20"],
    "cat": CAT_COLS,
    "xs": ([f"{c}_{s}" for c in XS_BASE for s in ["TS_MEAN", "TS_DEV", "TS_Z", "TS_RANK",
                                                  "TSG_MEAN", "TSG_DEV", "TSG_Z", "TSG_RANK"]]
           + ["SHARE_RET1_POS", "N_ALLOCS"]),
    "mr": ["MEAN_REV_5", "MEAN_REV_10", "MEAN_REV_20", "RET_MEAN_10",
           "ALLOC_AVERAGE_PERF_5", "ALLOC_AVERAGE_PERF_10", "ALLOC_AVERAGE_PERF_20"],
    # SV1_MISSING x FULL_REPORT_DAY is identically zero in train and test (nobody is missing on a
    # full-reporting day) so it is not included.
    "vol": ["SV1_REPORT_FRAC_TS", "FULL_REPORT_DAY", "ALLOC_SV1_RATE",
            "SV1_MISSING_x_ALLOCRATE", "SV1_SURPRISE"],
    "fac": ["COMMON_RET1", "IDIO_RET1", "COMMON_RET1_Z", "IDIO_RET1_Z"],
}
FEATURE_SETS = {
    "baseline": "base,dum,miss,roll",
    "s5": "base,dum,miss,roll,cat,xs,mr,vol",
    "final": "base,dum,miss,roll,cat,mr,vol,fac",
}
LGB_PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=200,
                  feature_fraction=0.8, lambda_l2=1.0, verbose=-1, seed=42, num_threads=8)
LGB_ROUNDS = 250


def feature_list(spec):
    cols = []
    for g in spec.split(","):
        cols += [c for c in GROUPS[g] if c not in cols]
    return cols


# --------------------------------------------------------------------------------- row-local
def add_local(df):
    df = df.copy()
    for g in [1, 2, 3, 4]:
        df[f"GROUP_{g}"] = (df["GROUP"] == g).astype(int)
    df["SV1_MISSING"] = df["SIGNED_VOLUME_1"].isna().astype(int)
    df["RET_MEAN_5"] = df[RET[:5]].mean(axis=1)
    df["RET_MEAN_10"] = df[RET[:10]].mean(axis=1)
    df["RET_MEAN_20"] = df[RET].mean(axis=1)
    df["RET_STD_20"] = df[RET].std(axis=1)
    for k in [5, 10, 20]:
        df[f"MEAN_REV_{k}"] = df["RET_1"] - df[f"RET_MEAN_{k}"]
    df["ALLOC_CODE"] = df["ALLOCATION"].str.extract(r"(\d+)")[0].astype(int)
    df["GROUP_CODE"] = df["GROUP"].astype(int)
    return df


# --------------------------------------------------------------------------------- per-date
def _relative(df, keys, cols, tag):
    g = df.groupby(keys, sort=False)
    for c in cols:
        m, s = g[c].transform("mean"), g[c].transform("std")
        df[f"{c}_{tag}_MEAN"] = m
        df[f"{c}_{tag}_DEV"] = df[c] - m
        df[f"{c}_{tag}_Z"] = (df[c] - m) / s.where(s > 0)
        df[f"{c}_{tag}_RANK"] = g[c].rank(pct=True)


def add_per_date(df):
    df = df.copy()
    g = df.groupby("TS", sort=False)
    df["N_ALLOCS"] = g["TS"].transform("size")
    df["_pos"] = (df["RET_1"] > 0).astype(float)
    df["SHARE_RET1_POS"] = g["_pos"].transform("mean")
    df.drop(columns="_pos", inplace=True)
    _relative(df, "TS", XS_BASE, "TS")
    _relative(df, ["TS", "GROUP"], XS_BASE, "TSG")
    for k in [5, 10, 20]:
        df[f"ALLOC_AVERAGE_PERF_{k}"] = g[f"RET_MEAN_{k}"].transform("mean")
    df["SV1_REPORT_FRAC_TS"] = 1.0 - g["SV1_MISSING"].transform("mean")
    df["FULL_REPORT_DAY"] = (df["SV1_REPORT_FRAC_TS"] >= 0.99).astype(int)
    return df


# --------------------------------------------------------------------------------- fold-dependent
def fit_factor_model(train_df, n_factors=N_FACTORS):
    piv = train_df.pivot(index="TS", columns="ALLOCATION", values="RET_1")
    std = piv.std().replace(0, np.nan).fillna(piv.std().median())
    Z = (piv / std).fillna(0.0).to_numpy()
    w, V = np.linalg.eigh((Z.T @ Z) / len(Z))
    idx = np.argsort(w)[::-1][:n_factors]
    return {"allocs": list(piv.columns), "std": std,
            "L": V[:, idx] * np.sqrt(np.maximum(w[idx], 0)), "k": n_factors}


def apply_factor_model(df, fm, ridge=1.0):
    df = df.copy()
    k, L = fm["k"], fm["L"]
    ai = df["ALLOCATION"].map(pd.Series(np.arange(len(fm["allocs"])), index=fm["allocs"])).to_numpy()
    std = df["ALLOCATION"].map(fm["std"]).to_numpy()
    r = df["RET_1"].to_numpy() / std
    ok = ~np.isnan(ai) & ~np.isnan(r)
    common_z = np.zeros(len(df))
    I = ridge * np.eye(k)
    for rows in df.groupby("TS", sort=False).indices.values():
        rows = np.asarray(rows)[ok[np.asarray(rows)]]
        if len(rows):
            Ls = L[ai[rows].astype(int)]
            common_z[rows] = Ls @ np.linalg.solve(Ls.T @ Ls + I, Ls.T @ r[rows])
    df["COMMON_RET1_Z"] = common_z
    df["IDIO_RET1_Z"] = np.nan_to_num(r) - common_z
    df["COMMON_RET1"] = common_z * np.nan_to_num(std)
    df["IDIO_RET1"] = df["RET_1"].fillna(0).to_numpy() - df["COMMON_RET1"].to_numpy()
    return df


def add_fold_dependent(train_df, other_df, n_factors=N_FACTORS):
    rate = 1.0 - train_df.groupby("ALLOCATION")["SV1_MISSING"].mean()
    fm = fit_factor_model(train_df, n_factors)
    out = []
    for d in (train_df, other_df):
        d = d.copy()
        d["ALLOC_SV1_RATE"] = d["ALLOCATION"].map(rate).fillna(rate.mean())
        d["SV1_MISSING_x_ALLOCRATE"] = d["SV1_MISSING"] * d["ALLOC_SV1_RATE"]
        d["SV1_SURPRISE"] = d["SV1_MISSING"] - (1.0 - d["ALLOC_SV1_RATE"])
        out.append(apply_factor_model(d, fm))
    return out


def build_features(train_df, other_df):
    """(train_feats, other_feats): every feature column, computed leak-free w.r.t. other_df."""
    return add_fold_dependent(add_per_date(add_local(train_df)), add_per_date(add_local(other_df)))


def to_frame(df, cols):
    X = df[cols].copy()
    num = [c for c in cols if c not in CAT_COLS]
    X[num] = X[num].astype(float).fillna(0.0)
    return X


# --------------------------------------------------------------------------------- model
def make_fit_predict(spec=FEATURE_SETS["final"], params=LGB_PARAMS, rounds=LGB_ROUNDS,
                     model="lgb", seed=42):
    """Returns fit_predict(train_df, train_y, val_df) -> P(target > 0) for qrt_prep.oof_predictions.
    train_y may be the raw target (sign is taken) or already 0/1. Score with threshold=0.5."""
    cols = feature_list(spec)
    cat_cols = [c for c in CAT_COLS if c in cols]

    def fit_predict(train_df, train_y, val_df):
        tr, va = build_features(train_df, val_df)
        y_sign = (np.asarray(train_y) > 0).astype(int)
        if model == "lgb":
            import lightgbm as lgb
            ds = lgb.Dataset(to_frame(tr, cols), label=y_sign,
                             categorical_feature=cat_cols or "auto", free_raw_data=False)
            m = lgb.train(dict(params, seed=seed), ds, num_boost_round=rounds)
            return m.predict(to_frame(va, cols))
        from catboost import CatBoostClassifier  # ties LightGBM here at ~6x the runtime
        m = CatBoostClassifier(iterations=300, learning_rate=0.015, depth=6, random_seed=seed,
                               verbose=False, thread_count=8, cat_features=cat_cols or None)
        m.fit(to_frame(tr, cols), y_sign)
        return m.predict_proba(to_frame(va, cols))[:, 1]

    return fit_predict


fit_predict = make_fit_predict()
