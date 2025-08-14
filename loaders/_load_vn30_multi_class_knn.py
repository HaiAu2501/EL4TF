from ._load_vn30_meta import _process_file, VN30
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np

def preprocess(symbol):
    df_train, df_test = _process_file(symbol, folder="multi_class_classification")

    # Encode label
    le = LabelEncoder()
    df_train["label_enc"] = le.fit_transform(df_train["label"])
    df_test["label_enc"] = le.transform(df_test["label"])
    
    # Tạo feature và target (dịch 1 ngày)
    def make_xy(df):
        X, y = [], []
        for t in range(len(df) - 1):
            feat = df.loc[t, ["open", "high", "low", "close", "label_enc"]].values
            target = df.loc[t + 1, "label_enc"]
            X.append(feat)
            y.append(target)
        return np.array(X), np.array(y)
    
    X_train, y_train = make_xy(df_train)
    X_test, y_test = make_xy(df_test)
    
    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
    }

