import numpy as np
import pandas as pd
from scipy.stats import norm

# Load Data
train_data = np.loadtxt("train.csv", delimiter=",")
test_data = np.loadtxt("test.csv", delimiter=",")

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Convert labels {0,1} to {-1,1} for Probit
y_train = 2 * y_train - 1
y_test = 2 * y_test - 1

#  (intercept)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

#  gradient of log-likelihood
def probit_gradient(w, X, y):
    wx = y * (X @ w)
    Phi = norm.cdf(wx)
    phi = norm.pdf(wx)
    grad = np.sum((y[:, np.newaxis] * phi[:, np.newaxis] / Phi[:, np.newaxis]) * X, axis=0) - w
    return grad

# Hessian of log-likelihood
def probit_hessian(w, X, y):
    wx = y * (X @ w)
    Phi = norm.cdf(wx)
    phi = norm.pdf(wx)
    r = (phi / Phi) * ((phi / Phi) + wx)  
    H = -np.sum((r[:, np.newaxis, np.newaxis] * (X[:, :, np.newaxis] @ X[:, np.newaxis, :])), axis=0) - np.eye(X.shape[1])
    return H

# Newton-Raphson Optimization
def train_probit_newton(X, y, max_iter=100, tol=1e-5):
    w = np.zeros(X.shape[1])  
    for _ in range(max_iter):
        grad = probit_gradient(w, X, y)
        H = probit_hessian(w, X, y)
        delta_w = np.linalg.solve(H, grad)  
        w_new = w - delta_w  
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w

# Train using Newton-Raphson
w_map_newton = train_probit_newton(X_train, y_train)

# Prediction function
def predict_probit(X, w):
    return np.sign(X @ w)

# Accuracy
def accuracy(X, y, w):
    preds = predict_probit(X, w)
    return np.mean(preds == y)

print(f"Test Accuracy (Newton-Raphson): {accuracy(X_test, y_test, w_map_newton):.4f}")