import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# dataset
train_file = "train.csv"
test_file = "test.csv"

# Column names 
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]

# train and test data
train_data = pd.read_csv(train_file, names=columns)
test_data = pd.read_csv(test_file, names=columns)

#  attributes
categorical_features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
encoder = OneHotEncoder(drop=None, sparse_output=False)
X_train = encoder.fit_transform(train_data[categorical_features])
X_test = encoder.transform(test_data[categorical_features])

# labels
label_mapping = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
y_train = train_data["label"].map(label_mapping).values
y_test = test_data["label"].map(label_mapping).values

# features and classes
num_classes = len(np.unique(y_train))
num_features = X_train.shape[1]

#  weights
w_init = np.zeros((num_features, num_classes))

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 

#  (MAP estimation)
def loss_function(W, X, y):
    W = W.reshape(num_features, num_classes)
    logits = X @ W  
    probs = softmax(logits) 

    # Negative log-likelihood
    log_likelihood = -np.sum(np.log(probs[np.arange(len(y)), y]))

    # Gaussian Prior (L2 Regularization)
    prior = 0.5 * np.sum(W**2)

    return log_likelihood + prior

# Gradient of the loss function
def gradient(W, X, y):
    W = W.reshape(num_features, num_classes)
    logits = X @ W
    probs = softmax(logits)
    
    
    Y_one_hot = np.eye(num_classes)[y]
    
    # Gradient calculation
    grad = X.T @ (probs - Y_one_hot) + W  

    return grad.ravel()

# L-BFGS optimizer
result = minimize(
    loss_function, w_init.ravel(), args=(X_train, y_train), method="L-BFGS-B", jac=gradient,
    options={"maxiter": 100, "ftol": 1e-5}
)

# optimized weights
W_opt = result.x.reshape(num_features, num_classes)

# Predict function
def predict(X, W):
    logits = X @ W
    probs = softmax(logits)
    return np.argmax(probs, axis=1) 

# accuracy
y_pred = predict(X_test, W_opt)
accuracy = np.mean(y_pred == y_test)

# test accuracy
print(f"Test Accuracy: {accuracy * 100:.2f}%")
