import numpy as np
import pandas as pd

RET = [f"RET_{i}" for i in range(1, 21)]
SV = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
BASE_FEATURES = RET + SV + ["MEDIAN_DAILY_TURNOVER"]


def load_raw(data_dir="data/raw"):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv", index_col="ROW_ID")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv", index_col="ROW_ID")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv", index_col="ROW_ID")
    return X_train, y_train, X_test


def make_folds(ts, n_splits=5, seed=0):
    """Group rows into folds by TS, not by row, so no date leaks across train/val.
    TS-grouped because a shared daily factor (~0.26 target correlation across allocations,
    see notebooks/eda.ipynb) means a row-random split would leak that day's outcome."""
    rng = np.random.default_rng(seed)
    unique_ts = np.array(ts.unique(), dtype=object)
    rng.shuffle(unique_ts)
    fold_of_ts = dict(zip(unique_ts, np.arange(len(unique_ts)) % n_splits))
    return ts.map(fold_of_ts).to_numpy()


DENSE_DAY_MIN_ALLOCS = 270


def dense_day_mask(df, min_allocs=DENSE_DAY_MIN_ALLOCS):
    """Rows on dates with >= min_allocs allocations present. The test block is dense (mean 266
    rows/day, 102 of 120 days >= 270) while half of train's dates are sparse (19-276 rows), and
    sparse days are easier (more one-directional). Scoring only dense days closes most of the
    CV-vs-leaderboard gap (~0.006); see notes/accuracy_ceiling.md."""
    return (df.groupby("TS")["TS"].transform("size") >= min_allocs).to_numpy()


def day_clustered_accuracy(hit, ts):
    """Mean accuracy plus a standard error that respects day-level clustering. Same-day targets
    share a common factor, so rows are not independent; the honest SE comes from treating each
    date's (row-weighted) accuracy as one observation."""
    hit = np.asarray(hit, dtype=float)
    day = pd.DataFrame({"ts": np.asarray(ts), "hit": hit}).groupby("ts")["hit"].agg(["mean", "size"])
    w = day["size"] / day["size"].sum()
    acc = float((w * day["mean"]).sum())
    n_days = len(day)
    se = float(np.sqrt((w**2 * (day["mean"] - acc) ** 2).sum() * n_days / max(n_days - 1, 1)))
    return acc, se


def cross_validate(df, y, fit_predict, n_splits=5, seed=0, dense_only=False):
    """fit_predict(train_df, train_y, val_df) -> val predictions (continuous, sign-scored).
    Returns per-fold accuracy; with dense_only=True each fold is scored on dense days only."""
    folds = make_folds(df["TS"], n_splits=n_splits, seed=seed)
    y_sign = (y > 0).astype(int)
    dense = dense_day_mask(df) if dense_only else np.ones(len(df), dtype=bool)
    accuracies = []
    for fold in range(n_splits):
        val_mask = folds == fold
        pred = fit_predict(df[~val_mask], y[~val_mask], df[val_mask])
        hit = (pred > 0).astype(int) == y_sign[val_mask]
        accuracies.append(hit[dense[val_mask]].mean())
    return np.array(accuracies)


def oof_predictions(df, y, fit_predict, n_splits=5, seed=0):
    """Out-of-fold continuous predictions for every row (for dense-day / clustered scoring)."""
    folds = make_folds(df["TS"], n_splits=n_splits, seed=seed)
    oof = np.zeros(len(df))
    for fold in range(n_splits):
        val_mask = folds == fold
        oof[val_mask] = fit_predict(df[~val_mask], y[~val_mask], df[val_mask])
    return oof


def report(df, y, oof, threshold=0.0):
    """Honest summary of an OOF prediction vector: all-day and dense-day accuracy, each with a
    day-clustered SE, plus the predicted-positive share (test base rate is ~0.507; models that
    predict 57-59% positive have been the ones that disappointed on the leaderboard)."""
    y_sign = (y > 0).astype(int)
    hit = ((oof > threshold).astype(int) == y_sign).astype(float)
    dense = dense_day_mask(df)
    acc_all, se_all = day_clustered_accuracy(hit, df["TS"])
    acc_dense, se_dense = day_clustered_accuracy(hit[dense], df["TS"].to_numpy()[dense])
    return {
        "acc_all": acc_all, "se_all": se_all,
        "acc_dense": acc_dense, "se_dense": se_dense,
        "pos_share": float((oof > threshold).mean()),
        "pos_share_dense": float((oof[dense] > threshold).mean()),
    }
