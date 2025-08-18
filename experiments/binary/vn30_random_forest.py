
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'loaders' can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from loaders._load_vn30_meta import VN30
from loaders._load_vn30_binary_hoang import preprocess


def to_numpy_features(X):
    """
    Convert features to numpy, clean NaN/Inf, clip extremes, cast to float64.
    """
    if hasattr(X, "values"):
        X = X.values

    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.clip(X, -1e10, 1e10)
    X = X.astype(np.float64)
    return X


def to_numpy_labels(y):
    """
    Convert labels to numpy int64 1D array.
    """
    if hasattr(y, "values"):
        y = y.values
    y = np.asarray(y).astype(np.int64).ravel()
    return y


def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=3,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_bal_accuracy": balanced_accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_bal_accuracy": balanced_accuracy_score(y_test, y_pred_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
        "feature_importances": getattr(model, "feature_importances_", None),
    }

    return model, metrics


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

            # Convert to numpy and clean
            X_train_np = to_numpy_features(X_train)
            X_val_np = to_numpy_features(X_val)
            X_test_np = to_numpy_features(X_test)
            y_train_np = to_numpy_labels(y_train)
            y_val_np = to_numpy_labels(y_val)
            y_test_np = to_numpy_labels(y_test)

            # Merge val into train for final training (optional)
            X_train_full = np.vstack([X_train_np, X_val_np])
            y_train_full = np.concatenate([y_train_np, y_val_np])

            # Train and evaluate
            model, metrics = train_and_evaluate_rf(X_train_full, y_train_full, X_test_np, y_test_np)

            # Persist key results
            all_results[symbol] = {
                "metrics": metrics,
            }

            # Print brief summary
            print(
                f"Train Acc: {metrics['train_accuracy']:.4f} | Test Acc: {metrics['test_accuracy']:.4f} | Test Bal Acc: {metrics['test_bal_accuracy']:.4f}"
            )

            # Top feature importances
            if metrics["feature_importances"] is not None:
                importances = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": metrics["feature_importances"],
                    }
                ).sort_values("importance", ascending=False)

                print("Top 10 features:")
                print(importances.head(10))

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
                    "Train_Acc": m["train_accuracy"],
                    "Test_Acc": m["test_accuracy"],
                    "Test_Bal_Acc": m["test_bal_accuracy"],
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        print("\nSummary across symbols:")
        print(summary_df)
        summary_df.to_csv("vn30_binary_random_forest_results.csv", index=False)
        print("Saved summary to vn30_binary_random_forest_results.csv")
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()


