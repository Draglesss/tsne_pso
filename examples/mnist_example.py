"""
Example of using TSNE-PSO on MNIST dataset.
This example demonstrates the high performance and quality of the implementation.
"""

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from tsne_pso import TSNEPSO
import time

# Load MNIST data
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype('float64')

# Normalize the data
X = X / 255.0

# Take a subset for demonstration
n_samples = 5000
rng = np.random.RandomState(42)
indices = rng.choice(X.shape[0], n_samples, replace=False)
X = X[indices]
y = y[indices]

# Initialize and run t-SNE with PSO
print("\nRunning t-SNE with PSO optimization...")
tsne = TSNEPSO(
    n_components=2,
    perplexity=30.0,
    n_particles=100,
    max_iter=1000,
    learning_rate=200.0,
    random_state=42,
    verbose=1
)

# Time the execution
start_time = time.time()
X_embedded = tsne.fit_transform(X)
end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.2f} seconds")

# Plot the results
plt.figure(figsize=(10, 10))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y.astype(int),
                     cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.title('MNIST digits embedded in 2D using t-SNE with PSO')
plt.show() 