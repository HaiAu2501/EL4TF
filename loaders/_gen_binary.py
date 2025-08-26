import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def generate_data(
    n_samples: int = 2000,
    season_period: int = 50,
    ar_phi: float = 0.7,
    ma_theta: float = 0.5,
    noise_sigma: float = 0.1,
    label_noise: float = 0.08,
    ratio: List[float] = [0.8, 0.0, 0.2],
    lag: int = 5,
    seed: int = 0,
    verbose: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Sinh dữ liệu time series stationary với 5 feature.
    X_t = concat[feature(t-lag), ..., feature(t-1)] (không dùng feature tại t).
    y_t = nhãn phi tuyến dựa trên feature tại t.
    Chia train/val/test theo thứ tự thời gian.
    """

    np.random.seed(seed)

    t = np.arange(n_samples)

    # Feature 1: Sinusoidal
    f1 = np.sin(2 * np.pi * t / season_period) + np.random.normal(0, noise_sigma, n_samples)

    # Feature 2: AR(1)
    f2 = np.zeros(n_samples)
    f2[0] = np.random.normal()
    for i in range(1, n_samples):
        f2[i] = ar_phi * f2[i-1] + np.random.normal(0, noise_sigma)

    # Feature 3: MA(1)
    eps3 = np.random.normal(0, noise_sigma, n_samples+1)
    f3 = eps3[1:] + ma_theta * eps3[:-1]

    # Feature 4: Sin + AR(1)
    f4 = np.zeros(n_samples)
    f4[0] = np.random.normal()
    for i in range(1, n_samples):
        f4[i] = np.sin(2 * np.pi * i / (season_period//2)) + ar_phi * f4[i-1] + np.random.normal(0, noise_sigma)

    # Feature 5: White noise
    f5 = np.random.uniform(-1, 1, n_samples)

    # --- Tạo nhãn dựa trên feature tại ngày t ---
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        val = (
            np.sin(f1[i]) +
            0.3 * (f2[i] ** 2) -
            0.5 * f3[i] * f4[i] +
            0.2 * f5[i] * f1[i] +
            np.cos(2*np.pi*i/30)
        )
        y[i] = 1 if val > 0 else 0

    # Thêm noise
    flip_idx = np.random.choice(n_samples, size=int(label_noise*n_samples), replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    # --- Build lagged feature matrix ---
    features = np.vstack([f1, f2, f3, f4, f5]).T
    X_lagged, y_lagged = [], []
    for i in range(lag, n_samples):
        # concat features từ t-lag,...,t-1
        window = features[i-lag:i].flatten()
        X_lagged.append(window)
        y_lagged.append(y[i])

    X_lagged = np.array(X_lagged)
    y_lagged = np.array(y_lagged)

    n_samples = len(y_lagged)

    # --- Split theo thứ tự thời gian ---
    n_train = int(ratio[0] * n_samples)
    n_val = int(ratio[1] * n_samples)

    X_train, y_train = X_lagged[:n_train], y_lagged[:n_train]
    X_val,   y_val   = X_lagged[n_train:n_train+n_val], y_lagged[n_train:n_train+n_val]
    X_test,  y_test  = X_lagged[n_train+n_val:], y_lagged[n_train+n_val:]

    if verbose:
        print(f"Generated dataset with {n_samples} samples.")
        print(f"Shapes: [{X_train.shape}, {X_val.shape}, {X_test.shape}]")

    dataset = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }

    return dataset
