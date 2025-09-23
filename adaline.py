import numpy as np

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

epoch = 0
while True:
    weight_update = np.zeros_like(weights)

    for i in range(X_bias.shape[0]):
        xi = X_bias[i]
        di = d[i]

        # Linear output
        yi = np.dot(weights, xi)

        # Error
        error = di - yi

        # Weight update
        delta_w = learning_rate * error * xi
        weight_update += delta_w

    # Average weight update
    weight_update /= X_bias.shape[0]

    # Check for convergence
    if np.all(np.abs(weight_update) < tolerance):
        print(f"Training completed in {epoch} epochs.")
        break

    # Update weights
    weights += weight_update
    epoch += 1

    # Safety stop
    if epoch > max_epochs:
        print("Max epochs reached. Training stopped.")
        break

print("Final weights (including bias):")
print(weights)

# Test the trained ADALINE
print("\nTesting on inputs:")
for xi, target in zip(X, d):
    xi_bias = np.insert(xi, 0, 1)  # add bias
    output = np.dot(weights, xi_bias)
    print(f"Input: {xi}, Target: {target}, Predicted (linear): {output:.3f}")
