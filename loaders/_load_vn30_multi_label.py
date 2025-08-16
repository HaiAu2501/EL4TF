import pandas as pd
import numpy as np
from ._load_vn30_meta import _process_file, VN30, TARGETS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_features(df: pd.DataFrame, lag: int = 30) -> pd.DataFrame:
    """
    Create essential technical features for multi-label classification
    
    Args:
        df: DataFrame with OHLCV data
        lag: Number of lag periods for technical indicators (default: 30)
    
    Returns:
        DataFrame with essential features only
    """
    df = df.copy()
    
    # Essential price-based features
    df['returns'] = df['close'].pct_change()
    df['price_range'] = (df['high'] - df['low']) / (df['close'] + 1e-8)  # Avoid division by zero
    
    # Key moving averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    # Price relative to moving averages (avoid division by zero)
    df['close_ma5_ratio'] = df['close'] / (df['ma5'] + 1e-8)
    df['close_ma20_ratio'] = df['close'] / (df['ma20'] + 1e-8)
    
    # Volume features (important for HighVol label)
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma5'] + 1e-8)
    
    # Momentum (important for Up label)
    df['momentum_5'] = df['close'] / (df['close'].shift(5) + 1e-8) - 1
    
    # RSI (important for trend detection)
    df['gain'] = np.where(df['returns'] > 0, df['returns'], 0)
    df['loss'] = np.where(df['returns'] < 0, -df['returns'], 0)
    df['avg_gain_14'] = df['gain'].rolling(window=14).mean()
    df['avg_loss_14'] = df['loss'].rolling(window=14).mean()
    df['rs_14'] = df['avg_gain_14'] / (df['avg_loss_14'] + 1e-8)  # Avoid division by zero
    df['rsi_14'] = 100 - (100 / (1 + df['rs_14']))
    
    # Dynamic lag features based on lag parameter
    lag_periods = [1, 5, 10, 15, 20, 25, 30]
    if lag < 30:
        # If lag < 30, use fewer periods
        lag_periods = [i for i in lag_periods if i <= lag]
    elif lag > 30:
        # If lag > 30, add more periods
        additional_periods = [i for i in range(35, lag + 1, 5)]
        lag_periods.extend(additional_periods)
    
    # Create lag features for close, volume, and returns
    for period in lag_periods:
        df[f'close_lag_{period}'] = df['close'].shift(period)
        df[f'volume_lag_{period}'] = df['volume'].shift(period)
        df[f'returns_lag_{period}'] = df['returns'].shift(period)
    
    # Additional robust features
    df['price_volatility'] = df['returns'].rolling(window=20).std()
    df['volume_volatility'] = df['volume'].rolling(window=20).std()
    
    # Normalize volume by price to make it more comparable across stocks
    df['volume_price_ratio'] = df['volume'] / (df['close'] + 1e-8)
    
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
    Preprocess data for multi-label classification with ensemble learning models
    
    Args:
        symbol: Stock symbol
        lag: Number of lag periods for features (default: 30)
        val_split: Validation set split ratio
        use_scaler: Whether to apply scaling
        random_state: Random seed for reproducibility
        verbose: Whether to print information
    
    Returns:
        dict: {
            "train": (X_train, y_train),
            "val": (X_val, y_val), 
            "test": (X_test, y_test),
            "scaler": fitted scaler object,
            "feature_names": list of feature names,
            "label_names": list of label names
        }
    """
    
    try:
        # Load data
        df_train, df_test = _process_file(symbol, folder="multi_label_classification")
        
        if verbose:
            print(f"Original train shape: {df_train.shape}")
            print(f"Original test shape: {df_test.shape}")
            print(f"Using lag: {lag}")
        
        # Create features with specified lag
        df_train_features = create_features(df_train, lag)
        df_test_features = create_features(df_test, lag)
        
        # Drop rows with NaN values (due to rolling windows and lags)
        df_train_features = df_train_features.dropna()
        df_test_features = df_test_features.dropna()
        
        if verbose:
            print(f"After feature creation - Train shape: {df_train_features.shape}")
            print(f"After feature creation - Test shape: {df_test_features.shape}")
        
        # Separate features and labels
        label_columns = ['Up', 'HighVol', 'BreakMA5']
        feature_columns = [col for col in df_train_features.columns 
                          if col not in ['time', 'open', 'high', 'low', 'close', 'volume'] + label_columns]
        
        X_train = df_train_features[feature_columns]
        y_train = df_train_features[label_columns]
        
        X_test = df_test_features[feature_columns]
        y_test = df_test_features[label_columns]
        
        # Ensure we have enough data
        if len(X_train) < 100 or len(X_test) < 50:
            raise ValueError(f"Insufficient data for {symbol}: train={len(X_train)}, test={len(X_test)}")
        
        # Split training data into train and validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_split, 
            random_state=random_state,
            stratify=None  # Multi-label data, can't stratify easily
        )
        
        if verbose:
            print(f"Final splits:")
            print(f"  Train: {X_train_split.shape}")
            print(f"  Val: {X_val.shape}")
            print(f"  Test: {X_test.shape}")
            print(f"  Labels: {y_train_split.shape}")
            print(f"  Feature names: {len(feature_columns)}")
        
        # Scaling
        scaler = None
        if use_scaler:
            scaler = StandardScaler()
            
            # Fit on training data only
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame to preserve column names
            X_train_final = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train_split.index)
            X_val_final = pd.DataFrame(X_val_scaled, columns=feature_columns, index=X_val.index)
            X_test_final = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
        else:
            X_train_final = X_train_split
            X_val_final = X_val
            X_test_final = X_test
        
        return {
            "train": (X_train_final, y_train_split),
            "val": (X_val_final, y_val),
            "test": (X_test_final, y_test),
            "scaler": scaler,
            "feature_names": feature_columns,
            "label_names": label_columns
        }
        
    except Exception as e:
        if verbose:
            print(f"Error in preprocess for {symbol}: {str(e)}")
        raise e
