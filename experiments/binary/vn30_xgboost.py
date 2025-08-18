import sys
from pathlib import Path

# Ensure project root is on sys.path so 'loaders' can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from loaders._load_vn30_meta import VN30
from loaders._load_vn30_binary_hoang import preprocess

from xgboost import XGBClassifier


def to_numpy_features(X):
    if hasattr(X, "values"):
        X = X.values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.clip(X, -1e10, 1e10)
    X = X.astype(np.float64)
    return X


def to_numpy_labels(y):
    if hasattr(y, "values"):
        y = y.values
    y = np.asarray(y).astype(np.int64).ravel()
    return y


def tune_threshold_on_val(model, X_val, y_val):
    # Use predicted probabilities and choose threshold that maximizes balanced accuracy
    proba = model.predict_proba(X_val)[:, 1]
    best_t, best_bal = 0.5, -1
    for t in np.linspace(0.2, 0.8, 61):
        preds = (proba >= t).astype(int)
        bal = balanced_accuracy_score(y_val, preds)
        if bal > best_bal:
            best_t, best_bal = t, bal
    return best_t, best_bal


def main():
    all_results = {}

    for symbol in VN30:
        print(f"\n{'=' * 50}")
        print(f"Processing {symbol}")
        print(f"{'=' * 50}")

        try:
            data = preprocess(symbol=symbol, lag=30, verbose=True)
            X_train, y_train = data["train"]
            X_val, y_val = data["val"]
            X_test, y_test = data["test"]

            feature_names = data["feature_names"]

            # Anti-leakage: forbid raw OHLCV and lag variants
            forbidden_exact = {"open", "high", "low", "close", "volume"}
            forbidden_prefixes = ("open_lag_", "high_lag_", "low_lag_", "close_lag_", "volume_lag_")
            present_exact = forbidden_exact.intersection(set(feature_names))
            present_prefix = [f for f in feature_names if f.startswith(forbidden_prefixes)]
            if present_exact or present_prefix:
                raise ValueError(
                    f"Leakage risk: found forbidden features: {sorted(list(present_exact)) + present_prefix}"
                )

            # Convert to numpy and clean
            X_train_np = to_numpy_features(X_train)
            X_val_np = to_numpy_features(X_val)
            X_test_np = to_numpy_features(X_test)
            y_train_np = to_numpy_labels(y_train)
            y_val_np = to_numpy_labels(y_val)
            y_test_np = to_numpy_labels(y_test)

            # XGBoost classifier
            model = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                tree_method="hist",
                n_jobs=-1,
            )

            # Train on train only, tune threshold on val
            model.fit(X_train_np, y_train_np)
            best_t, best_bal = tune_threshold_on_val(model, X_val_np, y_val_np)

            # Predictions
            proba_train = model.predict_proba(np.vstack([X_train_np, X_val_np]))[:, 1]
            y_train_full = np.concatenate([y_train_np, y_val_np])
            preds_train = (proba_train >= best_t).astype(int)

            proba_test = model.predict_proba(X_test_np)[:, 1]
            preds_test = (proba_test >= best_t).astype(int)

            metrics = {
                "threshold": best_t,
                "val_bal_acc": best_bal,
                "train_accuracy": accuracy_score(y_train_full, preds_train),
                "train_bal_accuracy": balanced_accuracy_score(y_train_full, preds_train),
                "test_accuracy": accuracy_score(y_test_np, preds_test),
                "test_bal_accuracy": balanced_accuracy_score(y_test_np, preds_test),
                "confusion_matrix": confusion_matrix(y_test_np, preds_test).tolist(),
                "classification_report": classification_report(y_test_np, preds_test, output_dict=True),
            }

            all_results[symbol] = {"metrics": metrics}

            print(
                f"Thresh={metrics['threshold']:.3f} | Val Bal Acc: {metrics['val_bal_acc']:.4f} | Train Acc: {metrics['train_accuracy']:.4f} | Test Acc: {metrics['test_accuracy']:.4f} | Test Bal Acc: {metrics['test_bal_accuracy']:.4f}"
            )

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary table
    if all_results:
        summary_rows = []
        for sym, res in all_results.items():
            m = res["metrics"]
            summary_rows.append(
                {
                    "Symbol": sym,
                    "Thresh": m["threshold"],
                    "Val_Bal_Acc": m["val_bal_acc"],
                    "Train_Acc": m["train_accuracy"],
                    "Test_Acc": m["test_accuracy"],
                    "Test_Bal_Acc": m["test_bal_accuracy"],
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        print("\nSummary across symbols:")
        print(summary_df)
        summary_df.to_csv("vn30_binary_xgboost_results.csv", index=False)
        print("Saved summary to vn30_binary_xgboost_results.csv")
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()


