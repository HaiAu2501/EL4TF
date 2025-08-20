import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Tuple
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


# --------- Các hàm kích hoạt nhanh ----------
def _tanh(x): return np.tanh(x)
def _relu(x): return np.maximum(x, 0.0)
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _sin(x): return np.sin(x)
def _gelu(x): 
    # x * Phi(x) ~ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715 x^3)))
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * (x**3))))
def _elu(x, alpha=1.0): 
    y = x.copy()
    mask = x < 0
    y[mask] = alpha * (np.exp(x[mask]) - 1.0)
    return y
def _identity(x): return x

_ACTS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": _tanh,
    "relu": _relu,
    "sigmoid": _sigmoid,
    "sin": _sin,
    "gelu": _gelu,
    "elu": _elu,
    "identity": _identity,
}


@dataclass
class RVFLConfig:
    n_nodes: int = 256
    n_layers: int = 1
    activation: str = "tanh"      # có thể: tanh/relu/gelu/sigmoid/sin/elu/identity
    alpha_grid: Optional[np.ndarray] = None  # grid cho RidgeCV
    n_splits_cv: int = 3           # TimeSeriesSplit cho RidgeCV
    random_state: Optional[int] = None


class _RVFLBase:
    """
    Một base RVFL: tạo n_layers random features, concat với X vào RidgeCV (đa đầu ra).
    """
    def __init__(self, cfg: RVFLConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.seed = seed if seed is not None else cfg.random_state
        self.W_: List[np.ndarray] = []
        self.b_: List[np.ndarray] = []
        self.models_: List[RidgeCV] = []  # 1 model cho mỗi target
        self.act_: Callable = _ACTS.get(cfg.activation, _tanh)
        self.in_dims_: List[int] = []
        self.feature_dim_: int = 0  # tổng chiều concat
        # grid alpha mặc định
        if self.cfg.alpha_grid is None:
            self.cfg.alpha_grid = np.logspace(-6, 4, 15)

    def _randn(self, shape):
        rng = np.random.RandomState(self.seed)
        return rng.randn(*shape)

    def _randu(self, shape, low=-1.0, high=1.0):
        rng = np.random.RandomState(self.seed + 1337 if self.seed is not None else None)
        return rng.uniform(low, high, size=shape)

    def _build_random_layers(self, X: np.ndarray):
        self.W_.clear(); self.b_.clear(); self.in_dims_.clear()
        H_list = []
        H = X
        for l in range(self.cfg.n_layers):
            in_dim = H.shape[1]
            self.in_dims_.append(in_dim)
            # He-like scaling đơn giản cho ổn định
            W = self._randn((in_dim, self.cfg.n_nodes)) / np.sqrt(max(1, in_dim))
            b = self._randu((self.cfg.n_nodes,))
            Z = H @ W + b
            H = self.act_(Z)
            H_list.append(H)
            # đổi seed nhẹ để mỗi layer thật sự khác
            if self.seed is not None:
                self.seed += 911

        # concat X với tất cả H_l
        Phi = X if len(H_list) == 0 else np.hstack([X] + H_list)
        self.feature_dim_ = Phi.shape[1]
        return Phi

    def _featurize(self, X: np.ndarray) -> np.ndarray:
        # dùng lại đã lưu W_, b_ để suy diễn
        H = X
        H_list = []
        for W, b in zip(self.W_, self.b_):
            Z = H @ W + b
            H = self.act_(Z)
            H_list.append(H)
        return X if len(H_list) == 0 else np.hstack([X] + H_list)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Huấn luyện multi-output bằng cách train riêng từng target với RidgeCV + TimeSeriesSplit.
        """
        # Khởi tạo & lưu ngẫu nhiên
        self.W_.clear(); self.b_.clear()
        H = X
        H_list = []
        rng_w = np.random.RandomState(self.seed)
        rng_b = np.random.RandomState(None if self.seed is None else self.seed + 1337)

        for l in range(self.cfg.n_layers):
            in_dim = H.shape[1]
            W = rng_w.randn(in_dim, self.cfg.n_nodes) / np.sqrt(max(1, in_dim))
            b = rng_b.uniform(-1.0, 1.0, size=(self.cfg.n_nodes,))
            Z = H @ W + b
            H = self.act_(Z)
            H_list.append(H)
            # xoay seed
            if self.seed is not None:
                rng_w = np.random.RandomState(self.seed + 911*(l+1))
                rng_b = np.random.RandomState(self.seed + 1337 + 577*(l+1))

            self.W_.append(W); self.b_.append(b)

        Phi = X if len(H_list) == 0 else np.hstack([X] + H_list)
        tscv = TimeSeriesSplit(n_splits=self.cfg.n_splits_cv)

        # Train từng target
        n_targets = Y.shape[1] if Y.ndim == 2 else 1
        if n_targets == 1:
            Y = Y.reshape(-1, 1)
        self.models_ = []
        for k in range(n_targets):
            model = RidgeCV(alphas=self.cfg.alpha_grid, cv=tscv, fit_intercept=True)
            model.fit(Phi, Y[:, k])
            self.models_.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self._featurize(X)
        preds = [m.predict(Phi) for m in self.models_]
        return np.vstack(preds).T  # (n_samples, n_targets)


class RVFLEnsemble:
    """
    RVFL Ensemble:
    - Tạo n_estimators base RVFL với seed khác nhau.
    - Huấn luyện bằng RidgeCV + TimeSeriesSplit.
    - Tính trọng số theo RMSE trên X_val/Y_val (per-target), rồi tổ hợp dự đoán có trọng số.
    """
    def __init__(
        self,
        n_nodes: int = 256,
        n_layers: int = 1,
        n_estimators: int = 5,
        activation: str = "tanh",
        n_splits_cv: int = 3,
        alpha_grid: Optional[np.ndarray] = None,
        base_seed: int = 42,
    ):
        self.cfg = RVFLConfig(
            n_nodes=n_nodes,
            n_layers=n_layers,
            activation=activation,
            n_splits_cv=n_splits_cv,
            alpha_grid=alpha_grid,
            random_state=None,
        )
        self.n_estimators = n_estimators
        self.base_seed = base_seed
        self.estimators_: List[_RVFLBase] = []
        self.val_rmse_: Optional[np.ndarray] = None     # shape (n_estimators, n_targets)
        self.weights_: Optional[np.ndarray] = None       # shape (n_estimators, n_targets)
        self.targets_: int = 0

    @staticmethod
    def _rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def fit_with_pack(self, pack: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Huấn luyện ensemble từ 'pack' (đã có X_train, Y_train, X_val, Y_val).
        """
        X_train, Y_train = pack["train"]
        X_val, Y_val = pack["val"]

        # Đảm bảo shape target là 2D
        if Y_train.ndim == 1: Y_train = Y_train.reshape(-1, 1)
        if Y_val.ndim == 1:   Y_val   = Y_val.reshape(-1, 1)
        self.targets_ = Y_train.shape[1]

        # Huấn luyện từng base với seed khác nhau
        self.estimators_.clear()
        for m in range(self.n_estimators):
            seed_m = self.base_seed + m * 9973
            est = _RVFLBase(self.cfg, seed=seed_m)
            est.fit(X_train, Y_train)
            self.estimators_.append(est)

        # Đánh giá trên validation và tạo weights
        val_preds = []
        for est in self.estimators_:
            val_preds.append(est.predict(X_val))
        val_preds = np.stack(val_preds, axis=0)  # (M, n_val, n_targets)

        # RMSE từng estimator, từng target
        rmses = np.zeros((self.n_estimators, self.targets_), dtype=float)
        for m in range(self.n_estimators):
            for t in range(self.targets_):
                rmses[m, t] = self._rmse(Y_val[:, t], val_preds[m, :, t])
        self.val_rmse_ = rmses

        # Trọng số: w = 1 / (rmse + eps) rồi chuẩn hóa theo cột (target)
        eps = 1e-12
        inv = 1.0 / (rmses + eps)
        # nếu có rmse NaN/inf thì fallback
        inv[~np.isfinite(inv)] = 0.0
        col_sums = inv.sum(axis=0, keepdims=True)
        # nếu cột nào sum=0 => đặt đều
        mask_zero = (col_sums == 0.0)
        if np.any(mask_zero):
            inv[:, mask_zero.flatten()] = 1.0
            col_sums = inv.sum(axis=0, keepdims=True)
        self.weights_ = inv / col_sums  # (M, T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán ensemble (weighted per-target). Nếu chưa có weights (chưa fit), ném lỗi.
        """
        if not self.estimators_:
            raise RuntimeError("Ensemble chưa được fit.")
        preds = np.stack([est.predict(X) for est in self.estimators_], axis=0)  # (M, n, T)

        if self.weights_ is None:
            # fallback: trung bình đều
            return preds.mean(axis=0)

        # nhân theo target: (M, n, T) * (M, T) -> (n, T)
        # broadcast weights (M, 1, T)
        W = self.weights_[:, None, :]  # (M, 1, T)
        P = (preds * W).sum(axis=0)    # (n, T)
        return P

    def evaluate_rmse(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Trả về RMSE từng target trên (X, Y).
        """
        if Y.ndim == 1: Y = Y.reshape(-1, 1)
        pred = self.predict(X)
        T = pred.shape[1]
        out = np.zeros(T, dtype=float)
        for t in range(T):
            out[t] = np.sqrt(mean_squared_error(Y[:, t], pred[:, t]))
        return out

    # tiện ích cho gói pack
    def fit_predict_with_pack(self, pack: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Fit bằng pack và trả về dự đoán cho cả val/test + rmse chẩn đoán.
        """
        self.fit_with_pack(pack)
        X_test, Y_test = pack["test"]
        scaler = pack["scaler"]["target"]

        pred_test = self.predict(X_test)
        Y_pred = scaler.inverse_transform(pred_test)
        Y_test = scaler.inverse_transform(Y_test)

        r2 = r2_score(Y_test, Y_pred)
        mape = mean_absolute_percentage_error(Y_test, Y_pred) * 100

        return r2, mape
