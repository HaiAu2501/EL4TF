import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# Giả định rằng bạn có các file/hàm này ở cùng một nơi
from ._load_vn30_meta import _process_file, VN30, TARGETS 

def preprocess(
    symbol: str,
    lag: int = 30,
    batch_size: int = 32,  # <-- THÊM THAM SỐ BATCH_SIZE
    lag_label: bool = False,
    use_scaler: bool = False,
    val: float = 0.1, 
    verbose: bool = False,
):
    """
    Preprocess data and return PyTorch DataLoaders directly.
    
    Returns:
        tuple: (train_loader, valid_loader, test_loader, scaler, classes)
    """

    df_train, df_test = _process_file(symbol, folder = "multi_class_classification")

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

    Y_train_full = df_train['label'].map(label_mapping).values
    Y_test = df_test['label'].map(label_mapping).values

    X_train_full = df_train.drop(columns=TARGETS + ['label']).values
    X_test = df_test.drop(columns=TARGETS + ['label']).values
    
    feature_scaler = None
    if use_scaler:
        feature_scaler = StandardScaler()
        X_train_full = feature_scaler.fit_transform(X_train_full)
        X_test = feature_scaler.transform(X_test)

    n_samples = X_train_full.shape[0]
    valid_size = int(n_samples * val)
    train_size = n_samples - valid_size

    X_train = X_train_full[:train_size]
    Y_train = Y_train_full[:train_size]
    X_val = X_train_full[train_size:]
    Y_val = Y_train_full[train_size:]

    if verbose:
        print(f"=== Preprocessing {symbol} ===")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    
    # 1. Chuyển đổi Numpy array thành PyTorch Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    # 2. Tạo TensorDataset từ các tensor
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    valid_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    # 3. Tạo DataLoader từ Dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    classes = [k for k, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    
    return train_loader, valid_loader, test_loader, feature_scaler, classes