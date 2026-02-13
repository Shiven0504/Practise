# ...existing code...
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

"""
ADALINE training for AND function (batch updates).
- Uses vectorized batch weight updates.
- Stops based on MSE change (tolerance) or max_epochs.
- Deterministic via local RNG when seed is provided (doesn't affect global np.random).
- Prints linear outputs and discrete predictions (threshold default 0.5).
"""

def train_adaline(
    X: np.ndarray,
    d: np.ndarray,
    learning_rate: float = 0.1,
    max_epochs: int = 1000,
    tolerance: float = 1e-6,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, List[float], int]:
    """Train ADALINE using batch LMS. Returns (weights, mse_history, epoch_reached)."""
    # Basic validation
    X = np.asarray(X)
    d = np.asarray(d)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if d.ndim != 1:
        d = d.ravel()
    if X.shape[0] != d.shape[0]:
        raise ValueError("Number of samples in X and d must match.")

    # Local RNG to avoid changing global random state
    rng = np.random.RandomState(seed)

    # Add bias input (column of ones)
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

    # Initialize weights randomly (bias + input dims)
    weights = rng.uniform(-0.5, 0.5, X_bias.shape[1])

    prev_mse = np.inf
    n_samples = X_bias.shape[0]
    mse_history: List[float] = []

    for epoch in range(1, max_epochs + 1):
        outputs = X_bias.dot(weights)          # shape (n_samples,)
        errors = d - outputs                   # shape (n_samples,)

        mse = float(np.mean(errors ** 2))
        mse_history.append(mse)

        # Batch weight update (gradient descent / LMS)
        weight_update = learning_rate * (X_bias.T.dot(errors)) / n_samples
        weights += weight_update

        if verbose and (epoch % 100 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d} MSE: {mse:.6e}")

        # Check for convergence (based on absolute or relative MSE change)
        if not np.isfinite(prev_mse):
            prev_mse = mse
            continue

        abs_change = abs(prev_mse - mse)
        rel_change = abs_change / (abs(prev_mse) + 1e-12)
        if abs_change < tolerance or rel_change < 1e-12:
            if verbose:
                print(f"Training converged in {epoch} epoch(s). MSE: {mse:.6e}")
            return weights, mse_history, epoch

        prev_mse = mse

    if verbose:
        print(f"Max epochs reached ({max_epochs}). Final MSE: {mse_history[-1]:.6e}")
    return weights, mse_history, max_epochs


def predict_adaline(weights: np.ndarray, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Return linear outputs and binary predictions (threshold is configurable) for inputs X."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    linear_outputs = X_bias.dot(weights)
    predictions = (linear_outputs >= threshold).astype(int)
    return linear_outputs, predictions


if __name__ == "__main__":
    # Input samples (AND function, 2 inputs)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Desired outputs
    d = np.array([0, 0, 0, 1])

    # Train (use local RNG via seed param so global RNG is not affected)
    weights, mse_history, epoch_reached = train_adaline(
        X, d,
        learning_rate=0.1,
        max_epochs=1000,
        tolerance=1e-6,
        seed=42,
        verbose=True
    )

    print("\nFinal weights (including bias):")
    print(weights)

    # Test the trained ADALINE (linear outputs and discrete predictions)
    linear_outputs, preds = predict_adaline(weights, X, threshold=0.5)
    print("\nTesting on inputs:")
    for xi, target, lin, p in zip(X, d, linear_outputs, preds):
        print(f"Input: {xi}, Target: {target}, Linear: {lin:.4f}, Predicted: {p}")

    # Optional: plot MSE history
    plt.figure(figsize=(6,3))
    plt.plot(mse_history, lw=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("ADALINE training MSE")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
# ...existing code...