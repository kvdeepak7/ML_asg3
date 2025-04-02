import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# Parameters
alpha = 2.0       
beta = 25.0       
w_true = np.array([-0.3, 0.5])  

# first data point (x1, y1)
np.random.seed(42)
x1 = np.random.uniform(-1, 1)
y1 = w_true[0] + w_true[1] * x1 + np.random.normal(0, 0.2)

#  matrix for single data point
Phi = np.array([[1, x1]])  # Shape (1,2)

# Posterior calculations
S_inv = (1/alpha) * np.eye(2) + beta * Phi.T @ Phi
S = np.linalg.inv(S_inv)
m = beta * S @ Phi.T * y1  


# 1. Posterior heatmap
w0_grid, w1_grid = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
positions = np.dstack((w0_grid, w1_grid))
posterior_pdf = multivariate_normal(m.flatten(), S).pdf(positions)

plt.figure(figsize=(8, 6))
plt.title("Posterior Distribution After 1 Data Point", fontsize=14)
sns.heatmap(posterior_pdf, cmap="viridis", 
           xticklabels=False, yticklabels=False,
           cbar_kws={"label": "Density"})

# Corrected coordinate mapping 
plt.scatter([50], [50 + (w_true[0] + 1) * 50],  
           c='red', marker='x', s=100, label="True weights")

plt.xlabel(r"$w_0$", fontsize=12)
plt.ylabel(r"$w_1$", fontsize=12)
plt.gca().invert_yaxis()
plt.legend()
plt.show()


# Sample from posterior
w_samples = np.random.multivariate_normal(m.flatten(), S, 20)

# Plot configuration
x_plot = np.linspace(-1, 1, 100)
plt.figure(figsize=(8, 6))
plt.title("Posterior Lines After 1 Observation", fontsize=14)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$y$", fontsize=12)

# Plot sampled lines
for w0, w1 in w_samples:
    plt.plot(x_plot, w0 + w1*x_plot, lw=1, alpha=0.3, color='blue')

# Plot ground truth line
plt.plot(x_plot, w_true[0] + w_true[1]*x_plot, 'k--', lw=2, label="True Line")

# Plot observation (red circle)
plt.scatter([x1], [y1], 
           s=150, 
           c='none',          
           edgecolor='red',    
           linewidths=2,       
           marker='o',       
           label="Observation $(x_1, y_1)$")

plt.grid(True)
plt.legend()
plt.show()