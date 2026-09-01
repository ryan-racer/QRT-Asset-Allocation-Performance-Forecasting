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


def cross_validate(df, y, fit_predict, n_splits=5, seed=0):
    """fit_predict(train_df, train_y, val_df) -> val predictions (continuous, sign-scored)."""
    folds = make_folds(df["TS"], n_splits=n_splits, seed=seed)
    y_sign = (y > 0).astype(int)
    accuracies = []
    for fold in range(n_splits):
        val_mask = folds == fold
        pred = fit_predict(df[~val_mask], y[~val_mask], df[val_mask])
        acc = ((pred > 0).astype(int) == y_sign[val_mask]).mean()
        accuracies.append(acc)
    return np.array(accuracies)
