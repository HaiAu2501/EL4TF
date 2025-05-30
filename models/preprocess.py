import os
import torch
import numpy as np
import pandas as pd
from typing import Literal
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

VN30 = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 
    'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX',
    'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
    'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
]

TARGETS = ['open', 'high', 'low', 'close'] # What we want to predict

def _make_features(
    df: pd.DataFrame,
    lag: int,
    calendar_feature: bool = False,
    rolling_feature: bool = False,
    technical_feature: bool = False,
    nonlinear_feature: bool = False,
    autocorr_feature: bool = False,
    trend_feature: bool = False,
) -> pd.DataFrame:

    parts = [df]

    # 1) Lag features
    lag_dict = {
        f'{feat}_lag_{i}': df[feat].shift(i)
        for feat in TARGETS
        for i in range(1, lag+1)
    }
    parts.append(pd.DataFrame(lag_dict, index=df.index))

    # 2) Calendar features
    if calendar_feature:
        cal = df['time']
        parts.append(pd.DataFrame({
            'dow_sin':   np.sin(2*np.pi*cal.dt.weekday      / 7),
            'dow_cos':   np.cos(2*np.pi*cal.dt.weekday      / 7),
            'dom_sin':   np.sin(2*np.pi*(cal.dt.day-1)     / 31),
            'dom_cos':   np.cos(2*np.pi*(cal.dt.day-1)     / 31),
            'month_sin': np.sin(2*np.pi*(cal.dt.month-1)   / 12),
            'month_cos': np.cos(2*np.pi*(cal.dt.month-1)   / 12),
            'woy_sin':   np.sin(2*np.pi*(cal.dt.isocalendar().week-1)/52),
            'woy_cos':   np.cos(2*np.pi*(cal.dt.isocalendar().week-1)/52),
        }, index=df.index))

    # 3) Rolling statistics
    if rolling_feature:
        stat = {}
        for feat in TARGETS:
            s = df[feat]
            stat[f'{feat}_roll_mean'] = s.rolling(lag).mean().shift(1)
            stat[f'{feat}_roll_std']  = s.rolling(lag).std().shift(1)
            stat[f'{feat}_roll_min']  = s.rolling(lag).min().shift(1)
            stat[f'{feat}_roll_max']  = s.rolling(lag).max().shift(1)
        parts.append(pd.DataFrame(stat, index=df.index))

    # 4) Technical indicators
    if technical_feature:
        close = df['close']
        tech = {}
        for n in (5,10,20):
            tech[f'sma_{n}'] = close.rolling(n).mean().shift(1)
            tech[f'ema_{n}'] = close.ewm(span=n,adjust=False).mean().shift(1)
        ema12 = close.ewm(span=12,adjust=False).mean()
        ema26 = close.ewm(span=26,adjust=False).mean()
        macd  = ema12 - ema26
        tech['macd']        = macd.shift(1)
        tech['macd_signal'] = macd.ewm(span=9,adjust=False).mean().shift(1)
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_g    = gain.rolling(14).mean()
        avg_l    = loss.rolling(14).mean()
        tech['rsi_14']      = (100 - 100/(1 + avg_g/avg_l)).shift(1)
        prev_c   = close.shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_c).abs()
        tr3 = (df['low']  - prev_c).abs()
        tr  = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
        tech['atr_14']      = tr.rolling(14).mean().shift(1)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        tech['bb_upper'] = (sma20 + 2*std20).shift(1)
        tech['bb_lower'] = (sma20 - 2*std20).shift(1)
        parts.append(pd.DataFrame(tech, index=df.index))

    # 5) Nonlinear features
    if nonlinear_feature:
        log_r = np.log(df['close'] / df['close'].shift(1))
        nl = {
            'log_return':   log_r.shift(1),
            'return_accel': (log_r - log_r.shift(1)).shift(1),
            'return_skew':  log_r.rolling(lag).skew().shift(1),
            'return_kurt':  log_r.rolling(lag).kurt().shift(1),
        }
        pr = df['close'].rolling(lag+1).apply(
            lambda x: np.sum(x[:-1] < x[-1]) / (len(x)-1),
            raw=True
        )
        nl['price_pct_rank'] = pr.shift(1)
        def fft_amp(x,k): return np.abs(np.fft.rfft(x))[k]
        amp1 = df['close'].rolling(lag).apply(lambda x: fft_amp(x,1), raw=True)
        amp2 = df['close'].rolling(lag).apply(lambda x: fft_amp(x,2), raw=True)
        nl['fft_amp1'] = amp1.shift(1)
        nl['fft_amp2'] = amp2.shift(1)
        parts.append(pd.DataFrame(nl, index=df.index))

    # 6) Autocorrelation features
    if autocorr_feature:
        ret = np.log(df['close']/df['close'].shift(1)).shift(1)
        ac = {}
        for k in (1,5,10):
            ac[f'acf_{k}'] = ret.rolling(window=lag+1).apply(
                lambda x: pd.Series(x).autocorr(k), raw=False
            )
        parts.append(pd.DataFrame(ac, index=df.index))

    # 7) Trend slope
    if trend_feature:
        def slope(x):
            y = x.values
            t = np.arange(len(y))
            return np.polyfit(t, y, 1)[0]
        tr = df['close'].rolling(lag).apply(slope, raw=False)
        parts.append(pd.DataFrame({'trend_slope': tr.shift(1)}, index=df.index))

    # Combine, replace inf, dropna, reset index
    df_full = pd.concat(parts, axis=1)
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_full.dropna(inplace=True)
    df_full.reset_index(drop=True, inplace=True)
    return df_full

def _process_file(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'vn30'
    df_train = pd.read_csv(data_dir / f'{symbol}_train.csv', parse_dates=['time'])
    df_test = pd.read_csv(data_dir / f'{symbol}_test.csv', parse_dates=['time'])
    return df_train, df_test

# For linear models
def preprocess_v1(
    symbol: str,
    lag: int = 30,
    val: float = 0.0, 
    calendar_feature: bool = True,
    rolling_feature: bool = True,
    technical_feature: bool = True,
    nonlinear_feature: bool = True,
    autocorr_feature: bool = True,
    trend_feature: bool = True,
    verbose: bool = False,
):
    """
    Preprocess the VN30 dataset for a given symbol.

    Parameters:
        symbol (str): The stock symbol to preprocess.
        lag (int): The number of lag features to create.
        val (float): The proportion of the training set to use for validation.
        calendar_feature (bool): Whether to include calendar features.
        rolling_feature (bool): Whether to include rolling features.
        technical_feature (bool): Whether to include technical features.
        nonlinear_feature (bool): Whether to include nonlinear features.
        autocorr_feature (bool): Whether to include autocorrelation features.
        trend_feature (bool): Whether to include trend features.
        verbose (bool): Whether to print preprocessing information.

    Returns:
        X_train (np.ndarray): The training features, shape (n_samples, n_features).
        Y_train (np.ndarray): The training targets, shape (n_samples, n_targets).
        X_val (np.ndarray): The validation features.
        Y_val (np.ndarray): The validation targets.
        X_test (np.ndarray): The test features.
        Y_test (np.ndarray): The test targets.
    """
    df_train, df_test = _process_file(symbol)

    df_train = df_train.drop(columns=['volume'])
    df_test = df_test.drop(columns=['volume'])

    df_train = _make_features(
        df_train,
        lag,
        calendar_feature,
        rolling_feature,
        technical_feature,
        nonlinear_feature,
        autocorr_feature,
        trend_feature,
    )
    df_test = _make_features(
        df_test,
        lag,
        calendar_feature,
        rolling_feature,
        technical_feature,
        nonlinear_feature,
        autocorr_feature,
        trend_feature,
    )

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
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    return {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test),
        "scaler": {
            "feature": feature_scaler, 
            "target": target_scaler, # When predicting, we need to inverse transform the target
        },
    }

# For CNN and RNN
def preprocess_v2(
	symbol: str,
    mode: Literal['cnn', 'rnn'],
	lag: int = 30,
	val: float = 0.1,
	batch_size: int = 32,
	verbose: bool = False,
) -> dict:
    """
    Preprocess the VN30 dataset for a given symbol and return DataLoader objects.

    Parameters:
        symbol (str): The stock symbol to preprocess.
        lag (int): The number of lag features to create.
        val (float): The proportion of the training set to use for validation.
        batch_size (int): The batch size for the DataLoader.
        verbose (bool): Whether to print preprocessing information.

    Returns:
        train_loader (DataLoader): The DataLoader for the training set.
        valid_loader (DataLoader): The DataLoader for the validation set.
        test_loader (DataLoader): The DataLoader for the test set.
    """
    df_train, df_test = _process_file(symbol)
    df_train = df_train[TARGETS].values
    df_test = df_test[TARGETS].values

    # Normalize the data
    scaler = StandardScaler()
    df_train = scaler.fit_transform(df_train)
    df_test = scaler.transform(df_test)

    X_full = []
    Y_full = []

    for i in range(len(df_train) - lag):
        X_full.append(df_train[i : i + lag]) # (window_size, n_dimensions)
        Y_full.append(df_train[i + lag]) # (n_dimensions,)

    X_full = np.stack(X_full) # (n_samples, window_size, n_dimensions)
    Y_full = np.stack(Y_full) # (n_samples, n_dimensions)

    X_test = []
    Y_test = []

    for i in range(len(df_test) - lag):
        X_test.append(df_test[i : i + lag])
        Y_test.append(df_test[i + lag])

    X_test = np.stack(X_test) # (n_samples, window_size, n_dimensions)
    Y_test = np.stack(Y_test) # (n_samples, n_dimensions)

    if mode == 'cnn':
        X_full = X_full.transpose(0, 2, 1) # (n_samples, n_dimensions, window_size)
        X_test = X_test.transpose(0, 2, 1) # (n_samples, n_dimensions, window_size)

    n_samples = X_full.shape[0]
    n_valid = int(n_samples * val)
    n_train = n_samples - n_valid

    X_train = X_full[:n_train]
    Y_train = Y_full[:n_train]
    X_valid = X_full[n_train:]
    Y_valid = Y_full[n_train:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"Train shape: {X_train.shape}, {Y_train.shape}")
        print(f"Valid shape: {X_valid.shape}, {Y_valid.shape}")

    return train_loader, valid_loader, test_loader, scaler
    
# For delta models
def preprocess_v3(
    symbol: str,
    lag: int = 30,
    val: float = 0.0,
    verbose: bool = False,
):
    """
    Xử lý dữ liệu cho dự đoán sai phân bậc 1.
    - X đầu ra có shape (n_samples, lag * n_dimensions)
    - Y đầu ra có shape (n_samples, n_dimensions)
    Trả về dict với các bộ (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) và scaler.
    """
    df_train, df_test = _process_file(symbol)
    df_train = df_train[TARGETS].values
    df_test = df_test[TARGETS].values

    # Normalize the data
    scaler = StandardScaler()
    df_train = scaler.fit_transform(df_train)
    df_test = scaler.transform(df_test)

    X_full = []
    Y_full = []

    for i in range(len(df_train) - lag):
        window = df_train[i : i + lag] # (lag, n_dimensions)
        diff = df_train[i + lag] - df_train[i + lag - 1] # (n_dimensions,)
        X_full.append(window.flatten())
        Y_full.append(diff)

    X_full = np.stack(X_full) # (n_samples, lag * n_dimensions)
    Y_full = np.stack(Y_full) # (n_samples, n_dimensions)

    X_test = []
    Y_test = []

    for i in range(len(df_test) - lag):
        window = df_test[i : i + lag]
        diff = df_test[i + lag] - df_test[i + lag - 1]
        X_test.append(window.flatten())
        Y_test.append(diff)

    X_test = np.stack(X_test) # (n_samples, lag * n_dimensions)
    Y_test = np.stack(Y_test) # (n_samples, n_dimensions)

    n_samples = X_full.shape[0]
    n_valid = int(n_samples * val)
    n_train = n_samples - n_valid
    
    X_train = X_full[:n_train]
    Y_train = Y_full[:n_train]
    X_valid = X_full[n_train:]
    Y_valid = Y_full[n_train:]

    if verbose:
        print(f"Train shape: {X_train.shape}, {Y_train.shape}")
        print(f"Valid shape: {X_valid.shape}, {Y_valid.shape}")

    return {
        "train": (X_train, Y_train),
        "val": (X_valid, Y_valid),
        "test": (X_test, Y_test),
        "scaler": scaler,
    }

# Example usage:
# if __name__ == "__main__":
#     preprocess_v3('ACB', verbose=True)