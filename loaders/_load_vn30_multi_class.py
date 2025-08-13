import pandas as pd
import numpy as np
from ._load_vn30_meta import _process_file, VN30, TARGETS
from sklearn.preprocessing import StandardScaler

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
    Preprocess data for multi-class classification with encoded labels [0, 1, 2, 3, 4].

    Returns:
        dict: {
            "train": (X_train, Y_train),
            "val": (X_val, Y_val),
            "test": (X_test, Y_test),
            "scaler": {"feature": ...},
            "classes": list of original class names
        }
    """

    df_train, df_test = _process_file(symbol, folder="multi_class_classification")

    # Remove volume
    df_train = df_train.drop(columns=['volume'])
    df_test = df_test.drop(columns=['volume'])

    # Class-to-index mapping (0-based)
    label_mapping = {
        'strong_down': 0,
        'weak_down': 1,
        'sideways': 2,
        'weak_up': 3,
        'strong_up': 4
    }

    # Create lag features for TARGETS
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

    # Add lag of labels as features (optional)
    if lag_label:
        df_train_label_encoded = df_train['label'].map(label_mapping)
        df_test_label_encoded = df_test['label'].map(label_mapping)
        for i in range(1, lag+1):
            lag_train[f'label_lag_{i}'] = df_train_label_encoded.shift(i)
            lag_test[f'label_lag_{i}'] = df_test_label_encoded.shift(i)

    # Combine lag features and drop NA
    df_train = pd.concat([df_train, pd.DataFrame(lag_train, index=df_train.index)], axis=1).dropna()
    df_test = pd.concat([df_test, pd.DataFrame(lag_test, index=df_test.index)], axis=1).dropna()

    # Drop 'time' column
    df_train = df_train.drop(columns=['time'])
    df_test = df_test.drop(columns=['time'])

    # Encode target labels
    Y_train_full = df_train['label'].map(label_mapping).values
    Y_test = df_test['label'].map(label_mapping).values

    # Extract features
    X_train_full = df_train.drop(columns=TARGETS + ['label']).values
    X_test = df_test.drop(columns=TARGETS + ['label']).values

    # Standardize
    feature_scaler = StandardScaler()
    X_train_full = feature_scaler.fit_transform(X_train_full)
    X_test = feature_scaler.transform(X_test)

    # Train / validation split
    n_samples = X_train_full.shape[0]
    valid_size = int(n_samples * val)
    train_size = n_samples - valid_size

    X_train = X_train_full[:train_size]
    Y_train = Y_train_full[:train_size]
    X_val = X_train_full[train_size:]
    Y_val = Y_train_full[train_size:]

    if verbose:
        print(f"=== Preprocessing {symbol} ===")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        values, counts = np.unique(Y_train, return_counts=True)
        for v, c in zip(values, counts):
            print(f"Class {v} ({list(label_mapping.keys())[v]}): {c} samples")

    return {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test),
        "scaler": {"feature": feature_scaler},
        "classes": [k for k, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    }
