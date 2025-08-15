import pandas as pd
import numpy as np
from ._load_vn30_meta import _process_file, VN30, TARGETS
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def preprocess(
    symbol: str,
    lag = 30,
    val = 0.0,
    lag_label = False,
    verbose = False
):
    df_train, df_test = _process_file(symbol, folder="binary")

    label_mapping = {"up": 1, "down": 0}

    # Map labels to binary values
    df_train['label'] = df_train['label'].map(label_mapping)
    df_test['label'] = df_test['label'].map(label_mapping)

    df_train = df_train.drop(columns=['volume'])
    df_test = df_test.drop(columns=['volume'])

    FEATURES = TARGETS
    if lag_label:
        FEATURES = TARGETS + ['label']

    lag_train = {
        f'{feat}_lag_{i}': df_train[feat].shift(i)
        for feat in FEATURES
        for i in range(1, lag + 1)
    }

    lag_test = {
        f'{feat}_lag_{i}': df_test[feat].shift(i)
        for feat in FEATURES
        for i in range(1, lag + 1)
    }

    df_train = pd.concat([df_train, pd.DataFrame(lag_train)], axis=1)
    df_train.dropna(inplace=True)  # Drop rows with NaN values after lagging
    df_test = pd.concat([df_test, pd.DataFrame(lag_test)], axis=1)
    df_test.dropna(inplace=True)  # Drop rows with NaN values after lagging

    X_train_full = df_train.drop(columns=TARGETS + ['label', 'time']).values
    Y_train_full = df_train['label'].values
    X_test = df_test.drop(columns=TARGETS + ['label', 'time']).values
    Y_test = df_test['label'].values

    n_samples = X_train_full.shape[0]
    valid_size = int(n_samples * val)
    train_size = n_samples - valid_size

    X_train = X_train_full[:train_size]
    Y_train = Y_train_full[:train_size]
    X_valid = X_train_full[train_size:]
    Y_valid = Y_train_full[train_size:]

    if verbose:
        print(f"=== Preprocessing {symbol} ===")
        print(f"Feature shapes: train {X_train.shape}, validation {X_valid.shape}, test {X_test.shape}")
        print(f"Label distribution: train {Counter(Y_train)}, validation {Counter(Y_valid)}, test {Counter(Y_test)}")

    return {
        "train": (X_train, Y_train),
        "valid": (X_valid, Y_valid),
        "test": (X_test, Y_test)
    }
