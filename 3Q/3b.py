import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

#  labels 
label_mapping = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
y_train = train_data["label"].map(label_mapping).values
y_test = test_data["label"].map(label_mapping).values

# Number of features
num_features = X_train.shape[1]
num_classes = len(label_mapping)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# (MAP estimation)
def loss_function(W, X, y):
    logits = X @ W
    probs = sigmoid(logits)
    
    # entropy loss
    log_likelihood = -np.sum(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
    
    # (L2 regularization)
    prior = 0.5 * np.sum(W**2)
    
    return log_likelihood + prior

#logistic regression
def gradient(W, X, y):
    logits = X @ W
    probs = sigmoid(logits)
    grad = X.T @ (probs - y) + W  
    return grad

#  4 separate binary logistic regression models
models = []
for class_label in range(num_classes):
    y_binary = (y_train == class_label).astype(int)  
    w_init = np.zeros(num_features)

    result = minimize(
        loss_function, w_init, args=(X_train, y_binary), method="L-BFGS-B", jac=gradient,
        options={"maxiter": 100, "ftol": 1e-5}
    )

    models.append(result.x)  


def predict(X, models):
    scores = np.array([sigmoid(X @ w) for w in models]) 
    return np.argmax(scores, axis=0)  

# test accuracy
y_pred = predict(X_test, models)
accuracy = np.mean(y_pred == y_test)

print(f"Test Accuracy (One-vs-Rest): {accuracy * 100:.2f}%")

#  (Softmax)
softmax_model = LogisticRegression(solver="lbfgs", max_iter=100)  # Removed 'multi_class' argument
softmax_model.fit(X_train, y_train)

# Predict on test data
y_pred_softmax = softmax_model.predict(X_test)

# accuracy
accuracy_softmax = accuracy_score(y_test, y_pred_softmax)

print(f"Test Accuracy (Multi-Class Logistic Regression): {accuracy_softmax * 100:.2f}%")

