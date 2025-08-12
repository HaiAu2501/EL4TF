import pandas as pd
from ._load_vn30_meta import _process_file, VN30, TARGETS
from sklearn.preprocessing import StandardScaler

def preprocess(
    symbol: str,
    lag: int = 30,
    val: float = 0.0, 
    verbose: bool = False,
):
    """
    Return preprocessed data for a given symbol.

    Parameters:
        symbol (str): The stock symbol to preprocess.
        lag (int): The number of lagged features to create.
        val (float): The proportion of the dataset to use for validation.
        verbose (bool): Whether to print detailed information.

    Returns:
        dict: A dictionary containing the preprocessed train, validation, and test sets.
            dict["train"]: (X_train, Y_train)
            dict["val"]: (X_val, Y_val)
            dict["test"]: (X_test, Y_test)
            dict["scaler"]: {
                "feature": feature_scaler,
                "target": target_scaler,
            }
    """

    df_train, df_test = _process_file(symbol)

    df_train = df_train.drop(columns=['volume'])
    df_test = df_test.drop(columns=['volume'])

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

    df_train = pd.concat([df_train, pd.DataFrame(lag_train, index=df_train.index)], axis=1)
    df_train.dropna(inplace=True)  # Drop rows with NaN values after lagging
    df_test = pd.concat([df_test, pd.DataFrame(lag_test, index=df_test.index)], axis=1)
    df_test.dropna(inplace=True)  # Drop rows with NaN values after lagging

    # Split features and target
    df_train = df_train.drop(columns=['time'])
    df_test = df_test.drop(columns=['time'])

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_full = df_train.drop(columns=TARGETS).values
    Y_train_full = df_train[TARGETS].values
    X_test = df_test.drop(columns=TARGETS).values
    Y_test = df_test[TARGETS].values

    # Normalize the data
    X_train_full = feature_scaler.fit_transform(X_train_full)
    Y_train_full = target_scaler.fit_transform(Y_train_full)
    X_test = feature_scaler.transform(X_test)
    Y_test = target_scaler.transform(Y_test)

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
        print(f"Target shapes in train: {Y_train.shape}, val: {Y_val.shape}, test: {Y_test.shape}")

    return {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test),
        "scaler": {
            "feature": feature_scaler, 
            "target": target_scaler, # When predicting, we need to inverse transform the target
        },
    }