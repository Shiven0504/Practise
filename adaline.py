# ...existing code...
import numpy as np

"""
ADALINE training for AND function (batch updates).
- Uses vectorized batch weight updates.
- Stops based on MSE change (tolerance) or max_epochs.
- Deterministic via fixed random seed.
- Prints linear outputs and discrete predictions (threshold 0.5).
"""

np.random.seed(42)

# Input samples (AND function, 2 inputs)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Desired outputs
d = np.array([0, 0, 0, 1])

# Parameters
learning_rate = 0.1
max_epochs = 1000
tolerance = 1e-6

# Add bias input (column of ones)
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize weights randomly (bias + 2 inputs)
weights = np.random.uniform(-0.5, 0.5, X_bias.shape[1])

prev_mse = np.inf
n_samples = X_bias.shape[0]
epoch_reached = 0

for epoch in range(1, max_epochs + 1):
    # Vectorized linear outputs for all samples
    outputs = X_bias.dot(weights)          # shape (n_samples,)
    errors = d - outputs                   # shape (n_samples,)

    # Batch weight update (gradient descent / LMS)
    weight_update = learning_rate * (X_bias.T.dot(errors)) / n_samples
    weights += weight_update

    mse = np.mean(errors ** 2)

    # Check for convergence (based on MSE change)
    if abs(prev_mse - mse) < tolerance:
        print(f"Training converged in {epoch} epoch(s). MSE: {mse:.6e}")
        epoch_reached = epoch
        break

    prev_mse = mse

else:
    # executed if loop didn't break
    print(f"Max epochs reached ({max_epochs}). Final MSE: {mse:.6e}")
    epoch_reached = max_epochs

print("\nFinal weights (including bias):")
print(weights)

# Test the trained ADALINE (linear outputs and discrete predictions)
print("\nTesting on inputs:")
for xi, target in zip(X, d):
    xi_bias = np.insert(xi, 0, 1)  # add bias
    linear_output = np.dot(weights, xi_bias)
    predicted = 1 if linear_output >= 0.5 else 0  # threshold for AND (0/1)
    print(f"Input: {xi}, Target: {target}, Linear: {linear_output:.4f}, Predicted: {predicted}")
# ...existing code...