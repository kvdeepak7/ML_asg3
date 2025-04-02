import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# Load Data
train_data = np.loadtxt("train.csv", delimiter=",")
test_data = np.loadtxt("test.csv", delimiter=",")

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# labels{0,1} to {-1,1} for Probit
y_train = 2 * y_train - 1
y_test = 2 * y_test - 1

#  (intercept)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

#  (Objective Function)
def probit_nll(w, X, y):
    wx = y * (X @ w)
    Phi = norm.cdf(wx)
    return -np.sum(np.log(Phi)) + 0.5 * np.dot(w, w)  

# Probit Gradient
def probit_gradient(w, X, y):
    wx = y * (X @ w)
    Phi = norm.cdf(wx)
    phi = norm.pdf(wx)
    grad = np.sum((y[:, np.newaxis] * phi[:, np.newaxis] / Phi[:, np.newaxis]) * X, axis=0) - w
    return -grad  

# L-BFGS
def train_probit(X, y, w_init, max_iter=100, tol=1e-5):
    result = minimize(probit_nll, w_init, args=(X, y), method="L-BFGS-B", jac=probit_gradient, 
                      options={"maxiter": max_iter, "ftol": tol})
    return result.x

# Train with zero-initialized weights
w_zero = np.zeros(X_train.shape[1])
w_map_zero = train_probit(X_train, y_train, w_zero)

# Train with random-initialized weights
w_random = np.random.randn(X_train.shape[1])
w_map_random = train_probit(X_train, y_train, w_random)

# Prediction function
def predict_probit(X, w):
    return np.sign(X @ w)

# accuracy
def accuracy(X, y, w):
    preds = predict_probit(X, w)
    return np.mean(preds == y)

print(f"Test Accuracy (Zero Init): {accuracy(X_test, y_test, w_map_zero):.4f}")
print(f"Test Accuracy (Random Init): {accuracy(X_test, y_test, w_map_random):.4f}")