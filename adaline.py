import numpy as np
import matplotlib.pyplot as plt

"""
ADALINE training for AND, OR, and XOR logic gates (batch updates).
- Uses vectorized batch weight updates.
- Stops based on MSE change (tolerance) or max_epochs.
- Deterministic via fixed random seed.
- Prints linear outputs and discrete predictions (threshold 0.5).
- Note: XOR is not linearly separable; ADALINE will not perfectly learn it.
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
    weight_history = []

    for epoch in range(1, max_epochs + 1):
        outputs = X_bias.dot(weights)          # shape (n_samples,)
        errors = d - outputs                   # shape (n_samples,)

        # Batch weight update (gradient descent / LMS)
        weight_update = learning_rate * (X_bias.T.dot(errors)) / n_samples
        weights += weight_update

        mse = np.mean(errors ** 2)
        mse_history.append(mse)
        weight_history.append(weights.copy())

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d} MSE: {mse:.6e}  Weights: {weights}")

        # Check for convergence (based on MSE change)
        if abs(prev_mse - mse) < tolerance:
            if verbose:
                print(f"Training converged in {epoch} epoch(s). MSE: {mse:.6e}")
            return weights, mse_history, epoch, weight_history

        prev_mse = mse

    if verbose:
        print(f"Max epochs reached ({max_epochs}). Final MSE: {mse:.6e}")
    return weights, mse_history, max_epochs, weight_history


def predict_adaline(weights, X):
    """Return linear outputs and binary predictions (threshold 0.5) for inputs X."""
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    linear_outputs = X_bias.dot(weights)
    predictions = (linear_outputs >= 0.5).astype(int)
    return linear_outputs, predictions


if __name__ == "__main__":
    # All 2-input combinations
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    GATES = {
        "AND": np.array([0, 0, 0, 1]),
        "OR":  np.array([0, 1, 1, 1]),
        "XOR": np.array([0, 1, 1, 0]),
    }

    fig, axes = plt.subplots(len(GATES), 2, figsize=(12, 4 * len(GATES)))

    for row, (gate_name, d) in enumerate(GATES.items()):
        print(f"\n{'='*50}")
        print(f"  Training ADALINE on {gate_name} gate")
        print(f"{'='*50}")

        weights, mse_history, epoch_reached, weight_history = train_adaline(
            X, d, learning_rate=0.1, max_epochs=1000, tolerance=1e-6, seed=42, verbose=True
        )

        print(f"\nFinal weights (including bias): {weights}")
        print(f"Converged / stopped at epoch: {epoch_reached}")

        linear_outputs, preds = predict_adaline(weights, X)
        print("\nTesting on inputs:")
        for xi, target, lin, p in zip(X, d, linear_outputs, preds):
            match = "OK" if p == target else "WRONG"
            print(f"  Input: {xi}  Target: {target}  Linear: {lin:.4f}  Predicted: {p}  [{match}]")

        # --- MSE plot ---
        weight_history_arr = np.array(weight_history)
        ax_mse, ax_w = axes[row]

        ax_mse.plot(mse_history, lw=1.5)
        ax_mse.set_xlabel("Epoch")
        ax_mse.set_ylabel("MSE")
        ax_mse.set_title(f"{gate_name} — Training MSE")
        ax_mse.grid(alpha=0.25)

        # --- Weight history plot ---
        labels = ["bias"] + [f"w{i}" for i in range(1, weight_history_arr.shape[1])]
        for idx, label in enumerate(labels):
            ax_w.plot(weight_history_arr[:, idx], lw=1.5, label=label)
        ax_w.set_xlabel("Epoch")
        ax_w.set_ylabel("Weight value")
        ax_w.set_title(f"{gate_name} — Weight History")
        ax_w.legend()
        ax_w.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()
