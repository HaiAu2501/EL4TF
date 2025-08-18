import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pathlib import Path
from typing import Tuple, Optional

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

def augment_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    # ----- Block bootstrap -----
    block_frac: float = 0.3,     # tỉ lệ mẫu tổng hợp từ các block
    block_len: int = 10,         # độ dài mỗi block (số hàng liên tiếp)
    block_noise_sigma: float = 0.01,  # cường độ nhiễu thêm vào block (theo std từng cột)
    # ----- Within-class time-window mixup -----
    mixup_frac: float = 0.3,     # tỉ lệ mẫu tạo bằng mixup (so với n_train)
    mixup_window: int = 30,      # chỉ trộn với điểm cách không quá ±window hàng
    mixup_alpha: float = 0.6,    # Beta(alpha, alpha) để lấy lambda
    # ----- Jitter độc lập -----
    jitter_frac: float = 0.2,    # tỉ lệ mẫu sao chép + jitter
    jitter_sigma: float = 0.02,  # cường độ jitter (theo std từng cột)
    # ----- Lựa chọn cột, cân bằng lớp -----
    use_numeric_only: bool = True,
    balance_classes: bool = False,  # cân bằng lớp thông qua block bootstrap + jitter
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tăng cường dữ liệu tabular có thứ tự thời gian, KHÔNG dùng thông tin từ test.
    - Không phá vỡ thứ tự trong từng block (block bootstrap).
    - Mixup chỉ giữa các điểm cùng nhãn và ở gần nhau theo thời gian.
    - Nhiễu được scale theo std từng cột, ước lượng trên train.

    Trả về:
        X_train_aug (DataFrame), y_train_aug (Series)
    """
    rng = np.random.default_rng(random_state)

    # Chỉ giữ cột số nếu muốn
    if use_numeric_only:
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        X = X_train[num_cols].copy()
    else:
        X = X_train.copy()
        num_cols = X.columns

    y = y_train.reset_index(drop=True)
    X = X.reset_index(drop=True)  # giữ thứ tự, nhưng index 0..n-1 cho tiện

    n, d = X.shape
    if n == 0 or d == 0:
        return X_train.copy(), y_train.copy()

    # Thống kê để scale nhiễu
    col_std = X.std(axis=0, ddof=1).replace(0, 1.0).astype(float).values  # tránh chia 0

    X_syn_list = []
    y_syn_list = []

    # -------------------------
    # 1) Block bootstrap
    # -------------------------
    n_blocks_rows = int(np.round(block_frac * n))
    if n_blocks_rows > 0 and block_len > 0:
        n_blocks = max(1, n_blocks_rows // block_len)
        for _ in range(n_blocks):
            if n <= block_len:
                start = 0
            else:
                start = int(rng.integers(0, n - block_len + 1))
            end = start + block_len
            block_X = X.iloc[start:end].to_numpy().copy()
            block_y = y.iloc[start:end].to_numpy().copy()

            if block_noise_sigma > 0:
                noise = rng.normal(loc=0.0, scale=block_noise_sigma, size=block_X.shape) * col_std
                block_X = block_X + noise

            X_syn_list.append(pd.DataFrame(block_X, columns=num_cols))
            y_syn_list.append(pd.Series(block_y))

    # -------------------------
    # 2) Within-class time-window mixup (cùng nhãn)
    # -------------------------
    n_mix = int(np.round(mixup_frac * n))
    if n_mix > 0:
        idx_by_class = {c: np.where(y.values == c)[0] for c in np.unique(y.values)}
        for c, idxs in idx_by_class.items():
            if idxs.size < 2:
                continue
            # số mẫu mixup cho lớp c ~ theo tỉ lệ kích thước lớp
            n_mix_c = int(np.round(n_mix * (idxs.size / n)))
            for _ in range(n_mix_c):
                i = int(rng.choice(idxs))
                # lấy j trong cửa sổ [i - W, i + W] giao với cùng lớp
                lo = max(0, i - mixup_window)
                hi = min(n - 1, i + mixup_window)
                window_idxs = idxs[(idxs >= lo) & (idxs <= hi)]
                if window_idxs.size <= 1:
                    # fallback: lấy trong cả lớp
                    window_idxs = idxs
                j = int(rng.choice(window_idxs))
                if j == i and idxs.size > 1:
                    # đảm bảo không trùng i nếu có thể
                    j = int(rng.choice(idxs[idxs != i]))

                lam = float(rng.beta(mixup_alpha, mixup_alpha))
                x_new = lam * X.iloc[i].values + (1 - lam) * X.iloc[j].values
                # cùng nhãn -> giữ nguyên nhãn c
                y_new = c

                X_syn_list.append(pd.DataFrame([x_new], columns=num_cols))
                y_syn_list.append(pd.Series([y_new]))

    # -------------------------
    # 3) Jitter các hàng ngẫu nhiên
    # -------------------------
    n_jitter = int(np.round(jitter_frac * n))
    if n_jitter > 0 and jitter_sigma > 0:
        sel = rng.integers(0, n, size=n_jitter)
        Xj = X.iloc[sel].to_numpy().copy()
        yj = y.iloc[sel].to_numpy().copy()
        noise = rng.normal(loc=0.0, scale=jitter_sigma, size=Xj.shape) * col_std
        Xj = Xj + noise

        X_syn_list.append(pd.DataFrame(Xj, columns=num_cols))
        y_syn_list.append(pd.Series(yj))

    # -------------------------
    # Cân bằng lớp nếu cần (after aug)
    # -------------------------
    if balance_classes:
        # tính lại tần suất
        y_all_temp = pd.concat([y] + y_syn_list, ignore_index=True) if y_syn_list else y.copy()
        counts = y_all_temp.value_counts()
        maj = counts.idxmax()
        target = counts.max()

        # oversample thêm cho các lớp thiểu số bằng block + jitter
        for c in counts.index:
            deficit = target - counts[c]
            if deficit <= 0:
                continue
            idxs = np.where(y.values == c)[0]
            if idxs.size == 0:
                continue
            # block nhỏ + jitter
            k = int(np.ceil(deficit / max(1, block_len)))
            for _ in range(k):
                start = int(rng.choice(idxs))
                end = min(n, start + block_len)
                block_X = X.iloc[start:end].to_numpy().copy()
                block_y = np.full(block_X.shape[0], c, dtype=y.dtype)

                noise = rng.normal(loc=0.0, scale=jitter_sigma, size=block_X.shape) * col_std
                block_X = block_X + noise

                X_syn_list.append(pd.DataFrame(block_X, columns=num_cols))
                y_syn_list.append(pd.Series(block_y))

    # Ghép tất cả & trả về
    if X_syn_list:
        X_aug = pd.concat([X] + X_syn_list, axis=0, ignore_index=True)
        y_aug = pd.concat([y] + y_syn_list, axis=0, ignore_index=True)
    else:
        X_aug, y_aug = X.copy(), y.copy()

    # Khôi phục các cột (nếu ban đầu có non-numeric bị loại)
    if use_numeric_only and set(num_cols) != set(X_train.columns):
        # chỉ trả về các cột đã được dùng (numeric). Nếu muốn giữ cột không số,
        # có thể merge theo index ở ngoài hàm.
        pass

    return X_aug, y_aug

# Example
# if __name__ == "__main__":
#     _process_file('ACB', folder='regression')