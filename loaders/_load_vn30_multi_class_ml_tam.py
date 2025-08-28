import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from ._load_vn30_meta import _process_file, VN30, TARGETS

def preprocess(
    symbol: str,
    lag: int = 30,
    lag_label: bool = False,
    use_scaler: bool = False,
    val: float = 0.1,
    verbose: bool = False,
):
    """
    Preprocess data and return Numpy arrays for ML models.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, classes)
    """

    df_train, df_test = _process_file(symbol, folder="multi_class_classification")

    df_train = df_train.drop(columns=['volume'])
    df_test = df_test.drop(columns=['volume'])

    label_mapping = {
        'strong_down': 0, 'weak_down': 1, 'sideways': 2,
        'weak_up': 3, 'strong_up': 4
    }

    lag_train = {f'{feat}_lag_{i}': df_train[feat].shift(i) for feat in TARGETS for i in range(1, lag+1)}
    lag_test = {f'{feat}_lag_{i}': df_test[feat].shift(i) for feat in TARGETS for i in range(1, lag+1)}

    if lag_label:
        df_train_label_encoded = df_train['label'].map(label_mapping)
        df_test_label_encoded = df_test['label'].map(label_mapping)
        for i in range(1, lag+1):
            lag_train[f'label_lag_{i}'] = df_train_label_encoded.shift(i)
            lag_test[f'label_lag_{i}'] = df_test_label_encoded.shift(i)

    df_train = pd.concat([df_train, pd.DataFrame(lag_train, index=df_train.index)], axis=1).dropna()
    df_test = pd.concat([df_test, pd.DataFrame(lag_test, index=df_test.index)], axis=1).dropna()

    df_train = df_train.drop(columns=['time'])
    df_test = df_test.drop(columns=['time'])

    y_train_full = df_train['label'].map(label_mapping).values
    y_test = df_test['label'].map(label_mapping).values

    X_train_full = df_train.drop(columns=TARGETS + ['label']).values
    X_test = df_test.drop(columns=TARGETS + ['label']).values

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X_train_full)
        X_test = scaler.transform(X_test)

    n_samples = X_train_full.shape[0]
    valid_size = int(n_samples * val)
    train_size = n_samples - valid_size

    X_train = X_train_full[:train_size]
    y_train = y_train_full[:train_size]
    X_valid = X_train_full[train_size:]
    y_valid = y_train_full[train_size:]

    # Gộp tập validation vào test cho ML
    X_test = np.concatenate([X_valid, X_test], axis=0)
    y_test = np.concatenate([y_valid, y_test], axis=0)

    if verbose:
        print(f"=== Preprocessing {symbol} ===")
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    classes = [k for k, _ in sorted(label_mapping.items(), key=lambda item: item[1])]

    return X_train, X_test, y_train, y_test, scaler, classes