import numpy as np
import pandas as pd

# Load Data
train_data = np.loadtxt("train.csv", delimiter=",")
test_data = np.loadtxt("test.csv", delimiter=",")

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# (intercept)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))

# gradient and Hessian
def compute_gradient_hessian(X, y, w):
    m = X.shape[0]
    y_pred = sigmoid(X @ w)
    error = y - y_pred

    # Gradient
    grad = X.T @ error - w  

    # Hessian
    S = np.diag((y_pred * (1 - y_pred)).flatten())  
    H = -(X.T @ S @ X) - np.eye(X.shape[1])  

    return grad, H

# Newton-Raphson method
def newton_raphson(X, y, w_init, max_iter=100, tol=1e-5):
    w = w_init
    for i in range(max_iter):
        grad, H = compute_gradient_hessian(X, y, w)
        w_new = w - np.linalg.inv(H) @ grad

        # convergence
        if np.linalg.norm(w_new - w) < tol:
            print(f"Converged in {i+1} iterations.")
            break

        w = w_new
    return w

# Train with zero-initialized weights
w_zero = np.zeros(X_train.shape[1])
w_map_zero = newton_raphson(X_train, y_train, w_zero)

# Train with random-initialized weights
w_random = np.random.randn(X_train.shape[1])
w_map_random = newton_raphson(X_train, y_train, w_random)

# Prediction function
def predict(X, w):
    return (sigmoid(X @ w) >= 0.5).astype(int)

# accuracy
def accuracy(X, y, w):
    preds = predict(X, w)
    return np.mean(preds == y)

print(f"Test Accuracy (Zero Init): {accuracy(X_test, y_test, w_map_zero):.4f}")
print(f"Test Accuracy (Random Init): {accuracy(X_test, y_test, w_map_random):.4f}")