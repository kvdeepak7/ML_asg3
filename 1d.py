import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# Parameters
alpha = 2.0
beta = 25.0
w_true = np.array([-0.3, 0.5])

# Generate 5 data points
np.random.seed(42)
x_data = np.random.uniform(-1, 1, 5)
y_data = w_true[0] + w_true[1] * x_data + np.random.normal(0, 0.2, 5)

# Design matrix for 5 data points
Phi = np.hstack([np.ones((5, 1)), x_data.reshape(-1, 1)])  # Shape (5,2)

# Posterior calculations
S_inv = (1/alpha) * np.eye(2) + beta * Phi.T @ Phi
S = np.linalg.inv(S_inv)
m = beta * S @ Phi.T @ y_data 


# 1. Posterior heatmap
w0_grid, w1_grid = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
positions = np.dstack((w0_grid, w1_grid))
posterior_pdf = multivariate_normal(m.flatten(), S).pdf(positions)

plt.figure(figsize=(8, 6))
plt.title("Posterior After 5 Data Points", fontsize=14)
sns.heatmap(posterior_pdf, cmap="viridis", xticklabels=False, yticklabels=False,
           cbar_kws={"label": "Density"})

# Plot true weights
w0_idx = int((w_true[0] + 1) * 50)  
w1_idx = int((1 - w_true[1]) * 50)   
plt.scatter(w0_idx, w1_idx, c='red', marker='x', s=100, label="True weights")
plt.xlabel(r"$w_0$", fontsize=12)
plt.ylabel(r"$w_1$", fontsize=12)
plt.gca().invert_yaxis()
plt.legend()
plt.show()


# 2. Sampled lines plot
w_samples = np.random.multivariate_normal(m.flatten(), S, 20)

x_plot = np.linspace(-1, 1, 100)
plt.figure(figsize=(8, 6))
plt.title("Lines Sampled from Posterior (5 Observations)", fontsize=14)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$y$", fontsize=12)

# Plot sampled lines
for w0, w1 in w_samples:
    plt.plot(x_plot, w0 + w1*x_plot, lw=1, alpha=0.3, color='blue')

# Plot ground truth and data points
plt.plot(x_plot, w_true[0] + w_true[1]*x_plot, 'k--', lw=2, label="True line")
plt.scatter(x_data, y_data, s=150, c='none', edgecolor='red', 
           linewidths=2, marker='o', label="Observations")
plt.grid(True)
plt.legend()
plt.show()