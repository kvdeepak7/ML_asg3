import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# Parameters
alpha = 2.0       
beta = 25.0       
w_true = [-0.3, 0.5]  

# Prior distribution: p(w) = N(w | 0, alpha*I)
prior_mean = np.array([0.0, 0.0])
prior_cov = alpha * np.eye(2)

# 1. Heatmap of the prior
w0_grid, w1_grid = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
prior_pdf = multivariate_normal(prior_mean, prior_cov).pdf(np.dstack((w0_grid, w1_grid)))

plt.figure(figsize=(10, 6))
plt.title("Heatmap of Prior Distribution $p(\mathbf{w})$", fontsize=14)
sns.heatmap(prior_pdf, cmap="viridis", xticklabels=False, yticklabels=False,
            cbar_kws={"label": "Density"}, vmin=0, vmax=np.max(prior_pdf))
plt.xlabel("$w_0$", fontsize=12)
plt.ylabel("$w_1$", fontsize=12)
plt.gca().invert_yaxis() 
plt.show()

# 2. Sample 20 instances of w from the prior
np.random.seed(42)  
w_samples = np.random.multivariate_normal(prior_mean, prior_cov, size=20)

# 3. Plot 20 lines y = w0 + w1x in [-1, 1]
x_plot = np.linspace(-1, 1, 100)
plt.figure(figsize=(10, 6))
plt.title("Lines Sampled from Prior", fontsize=14)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("$y$", fontsize=12)

for w0, w1 in w_samples:
    y_plot = w0 + w1 * x_plot
    plt.plot(x_plot, y_plot, lw=1, alpha=0.5)

# Plot the ground-truth line for comparison
plt.plot(x_plot, w_true[0] + w_true[1] * x_plot, 'k--', lw=2, label="Ground Truth")
plt.legend()
plt.grid(True)
plt.show()