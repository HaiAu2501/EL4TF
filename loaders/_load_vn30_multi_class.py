import pandas as pd
import numpy as np
from ._load_vn30_meta import _process_file, VN30, TARGETS
from sklearn.preprocessing import StandardScaler

def preprocess(
    symbol: str,
    lag: int = 30,
    lag_label: bool = False,
    val: float = 0.0, 
    verbose: bool = False,
):
    """
    Return preprocessed data for multi-class classification (ordinal labels).

    Parameters:
        symbol (str): The stock symbol to preprocess.
        lag (int): The number of lagged features to create.
        lag_label (bool): Whether to include lagged labels as features.
        val (float): The proportion of the dataset to use for validation.
        verbose (bool): Whether to print detailed information.

    Returns:
        dict: A dictionary containing the preprocessed train, validation, and test sets.
    """

    df_train, df_test = _process_file(symbol, folder="multi_class_classification")

    # Remove volume column
    df_train = df_train.drop(columns=['volume'])
    df_test = df_test.drop(columns=['volume'])

    # Ordinal label mapping
    label_mapping = {
        'strong_down': -2,
        'weak_down': -1,
        'sideways': 0,
        'weak_up': 1,
        'strong_up': 2
    }

    # Lag features for price data
    lag_train = {
        f'{feat}_lag_{i}': df_train[feat].shift(i)
        for feat in TARGETS
        for i in range(1, lag+1)
    }
    lag_test = {
        f'{feat}_lag_{i}': df_test[feat].shift(i)
        for feat in TARGETS
        for i in range(1, lag+1)
    }

    # Lagged labels as features (if enabled)
    if lag_label:
        df_train_label_encoded = df_train['label'].map(label_mapping)
        df_test_label_encoded = df_test['label'].map(label_mapping)
        for i in range(1, lag+1):
            lag_train[f'label_lag_{i}'] = df_train_label_encoded.shift(i)
            lag_test[f'label_lag_{i}'] = df_test_label_encoded.shift(i)

    # Add lagged features
    df_train = pd.concat([df_train, pd.DataFrame(lag_train, index=df_train.index)], axis=1)
    df_train.dropna(inplace=True)
    df_test = pd.concat([df_test, pd.DataFrame(lag_test, index=df_test.index)], axis=1)
    df_test.dropna(inplace=True)

    # Drop time column
    df_train = df_train.drop(columns=['time'])
    df_test = df_test.drop(columns=['time'])

    # Apply ordinal mapping to labels (output)
    Y_train_full = df_train['label'].map(label_mapping).values
    Y_test = df_test['label'].map(label_mapping).values

    # Prepare features
    feature_scaler = StandardScaler()
    X_train_full = df_train.drop(columns=TARGETS + ['label']).values
    X_test = df_test.drop(columns=TARGETS + ['label']).values

    # Normalize features
    X_train_full = feature_scaler.fit_transform(X_train_full)
    X_test = feature_scaler.transform(X_test)

    # Train/Val split
    n_samples = X_train_full.shape[0]
    valid_size = int(n_samples * val)
    train_size = n_samples - valid_size

    X_train = X_train_full[:train_size]
    Y_train = Y_train_full[:train_size]
    X_val = X_train_full[train_size:]
    Y_val = Y_train_full[train_size:]

    if verbose:
        print(f"=== Preprocessing {symbol} ===")
        print(f"Feature shapes in train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
        print(f"Label shapes in train: {Y_train.shape}, val: {Y_val.shape}, test: {Y_test.shape}")
        print(f"Class distribution in training:")
        unique, counts = np.unique(Y_train, return_counts=True)
        for cls_val, count in zip(unique, counts):
            print(f"  {cls_val}: {count}")

    return {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test),
        "scaler": {
            "feature": feature_scaler,
        },
        "classes": sorted(label_mapping.keys(), key=lambda k: label_mapping[k])
    }
