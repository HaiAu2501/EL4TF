import pandas as pd
import numpy as np
from ._load_vn30_meta import _process_file, VN30
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def create_features(df: pd.DataFrame, lag: int = 30) -> pd.DataFrame:
    """
    Create robust technical features for binary classification.

    Mirrors the multi-label pipeline for parity across tasks,
    with protections against division-by-zero and numerical instabilities.
    """
    df = df.copy()

    # Shift raw series to ensure features use ONLY past information (no lookahead)
    close_prev = df["close"].shift(1)
    high_prev = df["high"].shift(1)
    low_prev = df["low"].shift(1)
    volume_prev = df["volume"].shift(1)

    # Basic returns and price range (based on previous day's data)
    df["returns"] = df["close"].pct_change().shift(1)
    df["price_range"] = (high_prev - low_prev) / (close_prev + 1e-8)

    # Moving averages
    df["ma5"] = close_prev.rolling(window=5).mean()
    df["ma20"] = close_prev.rolling(window=20).mean()

    # Ratios vs moving averages (avoid div by zero)
    df["close_ma5_ratio"] = close_prev / (df["ma5"] + 1e-8)
    df["close_ma20_ratio"] = close_prev / (df["ma20"] + 1e-8)

    # Volume context
    df["volume_ma5"] = volume_prev.rolling(window=5).mean()
    df["volume_ratio"] = volume_prev / (df["volume_ma5"] + 1e-8)
    
    # Open gap (normalized open vs previous close) to avoid using raw open
    df["open_gap"] = df["open"] / (close_prev + 1e-8) - 1

    # Momentum
    df["momentum_5"] = close_prev / (close_prev.shift(5) + 1e-8) - 1

    # RSI-style features
    df["gain"] = np.where(df["returns"] > 0, df["returns"], 0)
    df["loss"] = np.where(df["returns"] < 0, -df["returns"], 0)
    df["avg_gain_14"] = df["gain"].rolling(window=14).mean()
    df["avg_loss_14"] = df["loss"].rolling(window=14).mean()
    df["rs_14"] = df["avg_gain_14"] / (df["avg_loss_14"] + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + df["rs_14"]))

    # Dynamic lag features (parity with multi-label). These are all from the past.
    lag_periods = [1, 5, 10, 15, 20, 25, 30]
    if lag < 30:
        lag_periods = [p for p in lag_periods if p <= lag]
    elif lag > 30:
        lag_periods.extend([p for p in range(35, lag + 1, 5)])

    for p in lag_periods:
        # Use only normalized/scale-free lags to avoid raw OHLC leakage
        df[f"returns_lag_{p}"] = df["returns"].shift(p)

    # Volatility features and normalization (based on past values)
    df["price_volatility"] = df["returns"].rolling(window=20).std()
    # Use normalized volume volatility instead of raw scale
    df["volume_returns"] = volume_prev.pct_change()
    df["volume_volatility"] = df["volume_returns"].rolling(window=20).std()
    df["volume_price_ratio"] = volume_prev / (close_prev + 1e-8)

    return df


def preprocess(
    symbol: str,
    lag: int = 30,
    val_split: float = 0.2,
    use_scaler: bool = True,
    random_state: int = 42,
    verbose: bool = False,
):
    """
    Preprocess binary classification data for ensemble models.

    Returns a dict with train/val/test splits, scaler, feature names, and label name.
    """
    try:
        # Load CSVs (both contain the target column 'label' as up/down)
        df_train, df_test = _process_file(symbol, folder="binary")

        # Map label strings to binary ints
        label_mapping = {"up": 1, "down": 0}
        if "label" not in df_train.columns or "label" not in df_test.columns:
            raise ValueError("Expected 'label' column in binary dataset")
        df_train["label"] = df_train["label"].map(label_mapping)
        df_test["label"] = df_test["label"].map(label_mapping)

        if verbose:
            print(f"Original train shape: {df_train.shape}")
            print(f"Original test shape: {df_test.shape}")
            print(f"Using lag: {lag}")

        # Feature engineering
        df_train_features = create_features(df_train, lag)
        df_test_features = create_features(df_test, lag)

        # Drop rows with NaNs created by rolling/lagging
        df_train_features = df_train_features.dropna()
        df_test_features = df_test_features.dropna()

        if verbose:
            print(f"After feature creation - Train shape: {df_train_features.shape}")
            print(f"After feature creation - Test shape: {df_test_features.shape}")

        # Select features (exclude raw OHLCV and target columns)
        label_column = "label"
        # Exclude ALL raw OHLCV to avoid leakage and non-stationary scales
        base_exclude = ["time", "open", "high", "low", "close", "volume", label_column]
        feature_columns = [c for c in df_train_features.columns if c not in base_exclude]

        X_train = df_train_features[feature_columns]
        y_train = df_train_features[label_column]
        X_test = df_test_features[feature_columns]
        y_test = df_test_features[label_column]

        # Guard against too-small datasets after lagging
        if len(X_train) < 100 or len(X_test) < 50:
            raise ValueError(
                f"Insufficient data after feature creation for {symbol}: train={len(X_train)}, test={len(X_test)}"
            )

        # Train/val split (stratify for binary target)
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=val_split, random_state=random_state, stratify=y_train
        )

        if verbose:
            print("Final splits:")
            print(f"  Train: {X_train_split.shape}")
            print(f"  Val:   {X_val.shape}")
            print(f"  Test:  {X_test.shape}")
            print(f"  Labels: train={y_train_split.shape}, val={y_val.shape}")
            print(f"  Feature names: {len(feature_columns)}")

        # Optional scaling
        scaler = None
        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            X_train_final = pd.DataFrame(
                X_train_scaled, columns=feature_columns, index=X_train_split.index
            )
            X_val_final = pd.DataFrame(X_val_scaled, columns=feature_columns, index=X_val.index)
            X_test_final = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
        else:
            X_train_final = X_train_split
            X_val_final = X_val
            X_test_final = X_test

        # Return y as pandas Series (parity with multi-label returning DataFrames for y)
        y_train_final = y_train_split.astype(int)
        y_val_final = y_val.astype(int)
        y_test_final = y_test.astype(int)

        return {
            "train": (X_train_final, y_train_final),
            "val": (X_val_final, y_val_final),
            "test": (X_test_final, y_test_final),
            "scaler": scaler,
            "feature_names": feature_columns,
            "label_name": label_column,
        }

    except Exception as e:
        if verbose:
            print(f"Error in preprocess (binary) for {symbol}: {str(e)}")
        raise e


