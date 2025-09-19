# utils/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(csv_path, target_column="target", test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_column]).values.astype("float32")
    y = df[target_column].values.astype("float32")
    print(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Split into {X_train.shape[0]} training and {X_test.shape[0]} test samples.")
    return X_train, X_test, y_train, y_test 
