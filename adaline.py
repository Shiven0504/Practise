import numpy as np
import matplotlib.pyplot as plt

"""
ADALINE training for AND function (batch updates).
- Uses vectorized batch weight updates.
- Stops based on MSE change (tolerance) or max_epochs.
- Deterministic via fixed random seed.
- Prints linear outputs and discrete predictions (threshold 0.5).
"""

def train_adaline(X, d, learning_rate=0.1, max_epochs=1000, tolerance=1e-6, seed=None, verbose=False):
    """Train ADALINE using batch LMS. Returns (weights, mse_history, epoch_reached)."""
    if seed is not None:
        np.random.seed(seed)

    # Add bias input (column of ones)
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

    # Initialize weights randomly (bias + input dims)
    weights = np.random.uniform(-0.5, 0.5, X_bias.shape[1])

    prev_mse = np.inf
    n_samples = X_bias.shape[0]
    mse_history = []

    for epoch in range(1, max_epochs + 1):
        outputs = X_bias.dot(weights)          # shape (n_samples,)
        errors = d - outputs                   # shape (n_samples,)

        # Batch weight update (gradient descent / LMS)
        weight_update = learning_rate * (X_bias.T.dot(errors)) / n_samples
        weights += weight_update

        mse = np.mean(errors ** 2)
        mse_history.append(mse)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d} MSE: {mse:.6e}")

        # Check for convergence (based on MSE change)
        if abs(prev_mse - mse) < tolerance:
            if verbose:
                print(f"Training converged in {epoch} epoch(s). MSE: {mse:.6e}")
            return weights, mse_history, epoch

        prev_mse = mse

    if verbose:
        print(f"Max epochs reached ({max_epochs}). Final MSE: {mse:.6e}")
    return weights, mse_history, max_epochs


def predict_adaline(weights, X):
    """Return linear outputs and binary predictions (threshold 0.5) for inputs X."""
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    linear_outputs = X_bias.dot(weights)
    predictions = (linear_outputs >= 0.5).astype(int)
    return linear_outputs, predictions


if __name__ == "__main__":
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

    # Train
    weights, mse_history, epoch_reached = train_adaline(X, d, learning_rate=0.1,
                                                       max_epochs=1000, tolerance=1e-6,
                                                       seed=42, verbose=True)

    print("\nFinal weights (including bias):")
    print(weights)
