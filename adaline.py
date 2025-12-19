# ...existing code...
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

"""
ADALINE training for AND function (batch updates).
- Uses vectorized batch weight updates.
- Stops based on MSE change (tolerance) or max_epochs.
- Deterministic via optional random seed.
- Prints linear outputs and discrete predictions (threshold 0.5).
"""

def train_adaline(
    X: np.ndarray,
    d: np.ndarray,
    learning_rate: float = 0.1,
    max_epochs: int = 1000,
    tolerance: float = 1e-6,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, List[float], int, bool]:
    """Train ADALINE using batch LMS.
    Returns (weights, mse_history, epoch_reached, converged_flag).
    """
    rng = np.random.default_rng(seed)

    # Add bias input (column of ones)
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

    # Initialize weights small random values (bias + input dims)
    weights = rng.normal(loc=0.0, scale=0.1, size=X_bias.shape[1])

    prev_mse = np.inf
    n_samples = X_bias.shape[0]
    mse_history: List[float] = []

    for epoch in range(1, max_epochs + 1):
        outputs = X_bias.dot(weights)          # shape (n_samples,)
        errors = d - outputs                   # shape (n_samples,)

        # Batch weight update (gradient descent / LMS)
        weight_update = learning_rate * (X_bias.T.dot(errors)) / n_samples
        weights += weight_update

        mse = float(np.mean(errors ** 2))
        mse_history.append(mse)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d} MSE: {mse:.6e}")

        # Check for convergence (based on MSE change)
        if abs(prev_mse - mse) < tolerance:
            if verbose:
                print(f"Training converged in {epoch} epoch(s). MSE: {mse:.6e}")
            return weights, mse_history, epoch, True

        prev_mse = mse

    if verbose:
        print(f"Max epochs reached ({max_epochs}). Final MSE: {mse:.6e}")
    return weights, mse_history, max_epochs, False


def predict_adaline(weights: np.ndarray, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Return linear outputs and binary predictions (threshold default 0.5) for inputs X."""
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

    # Train
    weights, mse_history, epoch_reached, converged = train_adaline(
        X, d,
        learning_rate=0.1,
        max_epochs=1000,
        tolerance=1e-6,
        seed=42,
        verbose=True
    )

    print("\nFinal weights (including bias):")
    print(weights)
    print(f"Converged: {converged} (epochs: {epoch_reached})")

    # Test the trained ADALINE (linear outputs and discrete predictions)
    linear_outputs, preds = predict_adaline(weights, X)
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