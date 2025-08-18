import pandas as pd
import numpy as np
from ._load_vn30_meta import _process_file, VN30, TARGETS, select_features, augment_features
from collections import Counter

def preprocess(
    symbol: str,
    lag: int = 30,
    val: float = 0.0,
    verbose: bool = False,
    use_rolling: bool = False,
    use_calendar: bool = False,
    feat_select: bool = False,
    feat_augment: bool = False
):
    df_train, df_test = _process_file(symbol, folder="binary")

    label_mapping = {"up": 1, "down": 0}
    for df in (df_train, df_test):
        df["label"] = df["label"].map(label_mapping)
        df.drop(columns=["volume"], inplace=True)

    test_size = len(df_test)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    for col in ["open", "high", "low", "close"]:
        df_all[f"{col}_shift"] = df_all[col].shift(1)
        df_all[f"{col}_diff"] = df_all[f"{col}_shift"].diff()

    diff_cols = [f"{c}_diff" for c in ["open", "high", "low", "close"]]
    lag_frames = {c.replace("_diff", f"_lag_0"): df_all[c] for c in diff_cols}
    for c in diff_cols:
        for i in range(1, lag + 1):
            lag_frames[c.replace("_diff", f"_lag_{i}")] = df_all[c].shift(i)

    lag_block = pd.DataFrame(lag_frames, index=df_all.index)
    df_all = pd.concat([df_all, lag_block], axis=1)

    if use_rolling:
        roll_feats = {}
        for c in diff_cols:
            roll = df_all[c].rolling(window=lag, min_periods=lag)
            roll_feats[f"{c}_roll_mean"] = roll.mean()
            roll_feats[f"{c}_roll_std"]  = roll.std()
            roll_feats[f"{c}_roll_min"]  = roll.min()
            roll_feats[f"{c}_roll_max"]  = roll.max()
        df_all = pd.concat([df_all, pd.DataFrame(roll_feats, index=df_all.index)], axis=1)

    if use_calendar:
        time = pd.to_datetime(df_all["time"])
        dow = time.dt.dayofweek
        dom = time.dt.day
        doy = time.dt.dayofyear
        df_all["sin_dow"] = np.sin(2 * np.pi * dow / 7);   df_all["cos_dow"] = np.cos(2 * np.pi * dow / 7)
        df_all["sin_dom"] = np.sin(2 * np.pi * dom / 31);  df_all["cos_dom"] = np.cos(2 * np.pi * dom / 31)
        df_all["sin_doy"] = np.sin(2 * np.pi * doy / 366); df_all["cos_doy"] = np.cos(2 * np.pi * doy / 366)

    y = df_all["label"]

    drop_cols = [
        "open", "high", "low", "close", "label", "time",  # thêm time
        "open_shift", "high_shift", "low_shift", "close_shift",  # bỏ cột phụ trợ
        "open_diff", "high_diff", "low_diff", "close_diff"
    ]
    drop_cols = [c for c in drop_cols if c in df_all.columns]
    X = df_all.drop(columns=drop_cols)

    X_train_full, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train_full, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    mask_tr = X_train_full.notna().all(axis=1)
    X_train_full, y_train_full = X_train_full[mask_tr], y_train_full[mask_tr]

    mask_te = X_test.notna().all(axis=1)
    X_test, y_test = X_test[mask_te], y_test[mask_te]

    if feat_select:
        X_train_full, y_train_full, X_test, y_test = select_features(
            X_train_full, y_train_full, X_test, y_test, frac=0.5,
        )

    if feat_augment:
        X_train_full, y_train_full = augment_features(
            X_train_full, y_train_full,
        )

    if val > 0:
        n_tr = int(len(X_train_full) * (1 - val))
        X_train, y_train = X_train_full.iloc[:n_tr], y_train_full.iloc[:n_tr]
        X_valid, y_valid = X_train_full.iloc[n_tr:], y_train_full.iloc[n_tr:]
    else:
        X_train, y_train = X_train_full, y_train_full
        X_valid, y_valid = None, None

    if verbose:
        print(f"=== Preprocessing {symbol} ===")
        print(f"Train: {X_train.shape} | Valid: {None if X_valid is None else X_valid.shape} | Test: {X_test.shape}")
        print(f"Label dist train: {Counter(y_train)}, test: {Counter(y_test)}")

    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid) if X_valid is not None else None,
        "test":  (X_test,  y_test)
    }

