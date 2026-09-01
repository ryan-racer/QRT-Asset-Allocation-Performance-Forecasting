import pandas as pd

from qrt_prep import RET, SV

GROUP_DUMMY_COLS = [f"GROUP_{g}" for g in [1, 2, 3, 4]]
MISSING_COLS = ["SV1_MISSING"]
ROLLING_COLS = ["RET_MEAN_5", "RET_MEAN_20", "RET_STD_20"]
CROSS_SECTIONAL_COLS = ["SAMEDAY_MEAN_RET1", "SAMEDAY_GROUP_MEAN_RET1"]
GROUP_INTERACTION_COLS = [f"RET1_x_GROUP_{g}" for g in [1, 2, 3, 4]]


def engineer(df):
    """Adds engineered columns to a copy of df. Safe to call once on train and once on
    test (or once on train+test concatenated) -- every column here is either allocation-local
    or aggregated within a single TS, and TS never overlaps between train and test, so there's
    no channel for test-time information to leak into train features or vice versa."""
    df = df.copy()

    for g in [1, 2, 3, 4]:
        df[f"GROUP_{g}"] = (df["GROUP"] == g).astype(int)

    df["SV1_MISSING"] = df["SIGNED_VOLUME_1"].isna().astype(int)

    df["RET_MEAN_5"] = df[RET[:5]].mean(axis=1)
    df["RET_MEAN_20"] = df[RET].mean(axis=1)
    df["RET_STD_20"] = df[RET].std(axis=1)

    df["SAMEDAY_MEAN_RET1"] = df.groupby("TS")["RET_1"].transform("mean")
    df["SAMEDAY_GROUP_MEAN_RET1"] = df.groupby(["TS", "GROUP"])["RET_1"].transform("mean")

    for g in [1, 2, 3, 4]:
        df[f"RET1_x_GROUP_{g}"] = df["RET_1"] * (df["GROUP"] == g)

    return df


def to_matrix(df, feature_cols):
    return df[feature_cols].fillna(0).to_numpy()
