import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parents[1]
sys.path.append(str(PROJECT_ROOT))
print(f"Project root: {PROJECT_ROOT}")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

from loaders._load_vn30_multi_label import preprocess
from loaders._load_vn30_meta import VN30, TARGETS

def validate_and_clean_data(X_train, X_test, y_train, y_test, symbol):
    """
    Validate and clean data to ensure it's suitable for training
    """
    print(f"Validating data for {symbol}...")
    
    # Convert pandas DataFrames to numpy arrays if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    print(f"Data types - X_train: {type(X_train)}, y_train: {type(y_train)}")
    print(f"Data types - X_test: {type(X_test)}, y_test: {type(y_test)}")
    
    # Check for infinity values
    if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
        print(f"Warning: Found infinity values in data for {symbol}")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Check for very large values
    if np.any(np.abs(X_train) > 1e10) or np.any(np.abs(X_test) > 1e10):
        print(f"Warning: Found very large values in data for {symbol}")
        X_train = np.clip(X_train, -1e10, 1e10)
        X_test = np.clip(X_test, -1e10, 1e10)
    
    # Check for NaN values
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        print(f"Warning: Found NaN values in data for {symbol}")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Ensure data is float64 for features and int64 for labels
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    print(f"After cleaning - X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"After cleaning - y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    
    # Final validation
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        raise ValueError(f"Still contains NaN values after cleaning for {symbol}")
    
    if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
        raise ValueError(f"Still contains infinity values after cleaning for {symbol}")
    
    return X_train, X_test, y_train, y_test

# Dictionary to store results for all symbols
all_results = {}

# Process each symbol
for symbol in VN30:
    print(f"\n{'='*50}")
    print(f"Processing symbol: {symbol}")
    print(f"{'='*50}")
    
    try:
        # Load data for multi-label classification
        data = preprocess(symbol=symbol, lag=30, verbose=True)
        
        # Extract data splits
        X_train, y_train = data["train"]
        X_test, y_test = data["test"]
        scaler = data["scaler"]
        feature_names = data["feature_names"]
        label_names = data["label_names"]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Feature names: {len(feature_names)}")
        print(f"Label names: {label_names}")
        
        # Validate and clean data
        X_train, X_test, y_train, y_test = validate_and_clean_data(
            X_train, X_test, y_train, y_test, symbol
        )
        
        # Initialize base Random Forest classifier
        base_rf = RandomForestClassifier(
            n_estimators=5,
            max_depth=8,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Wrap with MultiOutputClassifier for multi-label support
        model = MultiOutputClassifier(base_rf)
        
        print("Training Random Forest model with MultiOutputClassifier...")
        model.fit(X_train, y_train)
        
        # Make predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Calculate metrics for each label separately
        train_accuracies = []
        test_accuracies = []
        
        for i, label_name in enumerate(label_names):
            train_acc = balanced_accuracy_score(y_train[:, i], train_preds[:, i])
            test_acc = balanced_accuracy_score(y_test[:, i], test_preds[:, i])
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print(f"{label_name}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        
        # Average accuracy across all labels
        avg_train_acc = np.mean(train_accuracies)
        avg_test_acc = np.mean(test_accuracies)
        
        print(f"Average Training Accuracy: {avg_train_acc:.4f}")
        print(f"Average Test Accuracy: {avg_test_acc:.4f}")
        
        # Store results
        all_results[symbol] = {
            'train_accuracy': avg_train_acc,
            'test_accuracy': avg_test_acc,
            'train_predictions': train_preds,
            'test_predictions': test_preds,
            'model': model,
            'label_accuracies': dict(zip(label_names, test_accuracies))
        }
        
        # Feature importance (average across all outputs)
        feature_importance = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 most important features:")
        print(feature_importance_df.head(10))
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Summary of all results
print(f"\n{'='*60}")
print("SUMMARY OF ALL SYMBOLS")
print(f"{'='*60}")

if all_results:
    results_summary = []
    for symbol, results in all_results.items():
        results_summary.append({
            'Symbol': symbol,
            'Train_Accuracy': results['train_accuracy'],
            'Test_Accuracy': results['test_accuracy']
        })
    
    summary_df = pd.DataFrame(results_summary)
    print(summary_df)
    
    # Overall performance
    avg_train_acc = summary_df['Train_Accuracy'].mean()
    avg_test_acc = summary_df['Test_Accuracy'].mean()
    print(f"\nAverage Training Accuracy: {avg_train_acc:.4f}")
    print(f"Average Test Accuracy: {avg_test_acc:.4f}")
    
    print(f"\nProcessing completed for {len(all_results)} symbols.")
    
    # Save results to file
    summary_df.to_csv('vn30_random_forest_results.csv', index=False)
    print("Results saved to 'vn30_random_forest_results.csv'")
    
else:
    print("No symbols were processed successfully. Please check the errors above.")

    