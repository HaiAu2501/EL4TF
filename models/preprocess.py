import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

VN30 = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 
    'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX',
    'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
    'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
]

TARGETS = ['open', 'high', 'low', 'close', 'volume'] # What we want to predict

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


def preprocess_vn30(
    symbol: str,
    lag: int = 30, 
    calendar_feature: bool = False,
    rolling_feature: bool = False,
    technical_feature: bool = False,
    nonlinear_feature: bool = False,
    autocorr_feature: bool = False,
    trend_feature: bool = False,
    volvol_feature: bool = False,
    entropy_feature: bool = False,
):
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'vn30'
    df_train = pd.read_csv(data_dir / f'{symbol}_train.csv', parse_dates=['time'])
    df_test = pd.read_csv(data_dir / f'{symbol}_test.csv', parse_dates=['time'])

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
    drop_cols = ['time'] + TARGETS
    X_train = df_train.drop(columns=drop_cols).values
    Y_train = df_train[TARGETS].values
    X_test = df_test.drop(columns=drop_cols).values
    Y_test = df_test[TARGETS].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"=== Preprocessing {symbol} ===")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    preprocess_vn30('ACB', 30, True, True, True, True, True, True)