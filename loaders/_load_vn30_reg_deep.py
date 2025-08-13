import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ._load_vn30_meta import _process_file, VN30, TARGETS
from sklearn.preprocessing import StandardScaler
from typing import Literal

def preprocess(
	symbol: str,
    mode: Literal['tcn', 'lstm', 'tft'],
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

    if mode == 'tcn':
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