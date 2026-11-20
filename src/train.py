"""Train the Multiple Linear Regression model (Day 20, Phase 5).
Loads data/insurance_cleaned.csv, splits 80/20 (random_state=42),
trains by the normal equation and prints train/test metrics."""
import numpy as np
import pandas as pd

df = pd.read_csv("data/insurance_cleaned.csv")
X = df.drop(columns=["charges"]).values.astype(float)
y = df["charges"].values.astype(float)

rng = np.random.RandomState(42)
idx = rng.permutation(len(df))
test_idx, train_idx = idx[:267], idx[267:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

Xb_train = np.c_[np.ones(len(X_train)), X_train]
theta, *_ = np.linalg.lstsq(Xb_train, y_train, rcond=None)
np.savez("src/linear_regression_model.npz", theta=theta,
         features=np.array(df.drop(columns=["charges"]).columns))

def metrics(Xb, y_true):
    pred = Xb @ theta
    err = y_true - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = 1 - float(np.sum(err ** 2)) / float(np.sum((y_true - y_true.mean()) ** 2))
    return mae, rmse, r2

print("Train: MAE %.2f, RMSE %.2f, R2 %.4f" % metrics(Xb_train, y_train))
Xb_test = np.c_[np.ones(len(X_test)), X_test]
print("Test:  MAE %.2f, RMSE %.2f, R2 %.4f" % metrics(Xb_test, y_test))
print("Intercept:", round(float(theta[0]), 2))
