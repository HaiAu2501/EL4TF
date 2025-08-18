import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pathlib import Path

def _process_file(symbol: str, folder: str = 'regression') -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'vn30' / folder
    df_train = pd.read_csv(data_dir / f'{symbol}_train.csv', parse_dates=['time'])
    df_test = pd.read_csv(data_dir / f'{symbol}_test.csv', parse_dates=['time'])
    return df_train, df_test

VN30 = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 
    'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX',
    'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
    'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
]

TARGETS = ['open', 'high', 'low', 'close'] # What we want to predict

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    frac: float = 0.5
):
    """
    Lọc feature dựa trên khác biệt phân phối giữa hai lớp nhị phân (0/1).
    - Tính KS statistic, Cohen's d, JS divergence cho mỗi cột.
    - Chuẩn hoá và cộng lại thành 1 score.
    - Chọn top frac (mặc định 0.5) số cột theo score.

    Trả về: (X_train_new, y_train, X_test_new, y_test)
    """

    def cohens_d(a, b):
        m0, m1 = np.mean(a), np.mean(b)
        s0, s1 = np.var(a, ddof=1), np.var(b, ddof=1)
        n0, n1 = len(a), len(b)
        sp = np.sqrt(((n0 - 1) * s0 + (n1 - 1) * s1) / (n0 + n1 - 2))
        return 0 if sp == 0 else (m1 - m0) / sp

    def js_divergence(a, b, bins=30):
        pa, _ = np.histogram(a, bins=bins, density=True)
        pb, _ = np.histogram(b, bins=bins, density=True)
        pa = pa / pa.sum()
        pb = pb / pb.sum()
        m = 0.5 * (pa + pb)
        kl = lambda p, q: np.sum(np.where(p > 0, p * (np.log(p + 1e-12) - np.log(q + 1e-12)), 0))
        return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)

    # Tách lớp
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    scores = {}
    for col in X_train.columns:
        a = X0[col].values
        b = X1[col].values

        # KS test
        ks_stat = ks_2samp(a, b).statistic
        # Cohen's d
        d_val = abs(cohens_d(a, b))
        # JS divergence
        js_val = js_divergence(a, b)

        # Score = trung bình chuẩn hoá
        raw = np.array([ks_stat, d_val, js_val])
        score = np.mean(raw / (raw.max() + 1e-12))  # chuẩn hoá theo giá trị max trong vector
        scores[col] = score

    # Sắp xếp theo score giảm dần
    sorted_cols = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    n_keep = max(1, int(len(sorted_cols) * frac))
    keep_cols = sorted_cols[:n_keep]

    return X_train[keep_cols].copy(), y_train, X_test[keep_cols].copy(), y_test

def augment_features():
    pass

# Example
# if __name__ == "__main__":
#     _process_file('ACB', folder='regression')