"""Autocorrelator (Autoassociative) Neural Network Implementation.

Implements an autocorrelator/autoassociative memory network that stores
and recalls bipolar patterns. Uses outer product learning rule (Hebbian)
to compute the weight matrix and thresholding for pattern recall.

This is related to Hopfield networks but specifically designed for
pattern storage and retrieval via autocorrelation.

Typical usage:
    >>> from autocorrelator import Autocorrelator
    >>> import numpy as np
    >>> patterns = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])
    >>> net = Autocorrelator(n_neurons=4)
    >>> net.train(patterns)
    >>> recalled = net.recall(np.array([1, -1, 1, 1]))  # noisy input
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Autocorrelator:
    """Autoassociative memory network using Hebbian learning.

    Stores bipolar patterns ({-1, +1}) and recalls the closest
    stored pattern given a (possibly corrupted) input.

    Attributes:
        n_neurons: Number of neurons in the network.
        weights: Weight matrix of shape (n_neurons, n_neurons).
        n_patterns: Number of stored patterns.
    """

    def __init__(self, n_neurons: int) -> None:
        """Initialize the autocorrelator network.

        Args:
            n_neurons: Number of neurons (must match pattern length).

        Raises:
            ValueError: If n_neurons < 1.
        """
        if n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")

        self.n_neurons = n_neurons
        self.weights: npt.NDArray[np.float64] = np.zeros(
            (n_neurons, n_neurons), dtype=np.float64
        )
        self.n_patterns: int = 0

    def train(self, patterns: npt.NDArray[np.float64]) -> None:
        """Store patterns using the outer product (Hebbian) rule.

        Computes: W = (1/P) * Σ(p_i * p_i^T) with zero diagonal.

        Args:
            patterns: Matrix of bipolar patterns, shape (n_patterns, n_neurons).
                Each element should be +1 or -1.

        Raises:
            ValueError: If pattern dimension doesn't match n_neurons.
        """
        if patterns.ndim == 1:
            patterns = patterns.reshape(1, -1)

        if patterns.shape[1] != self.n_neurons:
            raise ValueError(
                f"Pattern length {patterns.shape[1]} != n_neurons {self.n_neurons}"
            )

        self.n_patterns = patterns.shape[0]

        # Outer product rule
        self.weights = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float64)
        for p in patterns:
            self.weights += np.outer(p, p)

        # Normalize and zero diagonal (no self-connections)
        self.weights /= self.n_patterns
        np.fill_diagonal(self.weights, 0)

    def recall(
        self,
        pattern: npt.NDArray[np.float64],
        max_iterations: int = 100,
        mode: str = "synchronous",
    ) -> npt.NDArray[np.int_]:
        """Recall stored pattern from a (possibly noisy) input.

        Args:
            pattern: Input pattern of shape (n_neurons,), bipolar values.
            max_iterations: Maximum update iterations before giving up.
            mode: Update mode — 'synchronous' (all at once) or
                'asynchronous' (one neuron at a time, random order).

        Returns:
            Recalled bipolar pattern of shape (n_neurons,).

        Raises:
            ValueError: If pattern length doesn't match n_neurons.
            ValueError: If mode is not 'synchronous' or 'asynchronous'.
        """
        if len(pattern) != self.n_neurons:
            raise ValueError(
                f"Pattern length {len(pattern)} != n_neurons {self.n_neurons}"
            )
        if mode not in ("synchronous", "asynchronous"):
            raise ValueError(f"mode must be 'synchronous' or 'asynchronous', got '{mode}'")

        state = np.array(pattern, dtype=np.float64).copy()

        for _ in range(max_iterations):
            prev_state = state.copy()

            if mode == "synchronous":
                net_input = self.weights @ state
                state = np.where(net_input >= 0, 1, -1).astype(np.float64)
            else:
                # Asynchronous: update neurons in random order
                order = np.random.permutation(self.n_neurons)
                for i in order:
                    net_input_i = self.weights[i] @ state
                    state[i] = 1.0 if net_input_i >= 0 else -1.0

            # Check convergence
            if np.array_equal(state, prev_state):
                break

        return state.astype(np.int_)

    def energy(self, state: npt.NDArray[np.float64]) -> float:
        """Compute the energy of a network state.

        E = -0.5 * s^T * W * s

        Lower energy indicates a more stable state (closer to stored pattern).

        Args:
            state: Network state of shape (n_neurons,).

        Returns:
            Energy value (scalar). Stored patterns are energy minima.
        """
        return float(-0.5 * state @ self.weights @ state)

    def capacity(self) -> float:
        """Estimate the theoretical storage capacity.

        For a Hopfield-like network, the reliable capacity is
        approximately 0.138 * n_neurons patterns.

        Returns:
            Estimated maximum number of reliably storable patterns.
        """
        return 0.138 * self.n_neurons

    def test_recall(
        self, patterns: npt.NDArray[np.float64], noise_level: float = 0.1
    ) -> float:
        """Test recall accuracy with noisy inputs.

        Args:
            patterns: Original stored patterns, shape (n_patterns, n_neurons).
            noise_level: Fraction of bits to flip (0.0 = no noise, 1.0 = all flipped).

        Returns:
            Recall accuracy as fraction of perfectly recalled patterns.
        """
        if patterns.ndim == 1:
            patterns = patterns.reshape(1, -1)

        correct = 0
        n_flip = max(1, int(self.n_neurons * noise_level))

        for p in patterns:
            noisy = p.copy()
            flip_idx = np.random.choice(self.n_neurons, size=n_flip, replace=False)
            noisy[flip_idx] *= -1

            recalled = self.recall(noisy)
            if np.array_equal(recalled, p.astype(np.int_)):
                correct += 1

        return correct / len(patterns)


if __name__ == "__main__":
    np.random.seed(42)

    # Store 3 bipolar patterns of length 8
    patterns = np.array(
        [
            [1, 1, -1, -1, 1, 1, -1, -1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float64,
    )

    net = Autocorrelator(n_neurons=8)
    net.train(patterns)

    print(f"Stored {net.n_patterns} patterns")
    print(f"Theoretical capacity: {net.capacity():.1f} patterns\n")

    # Test recall with noise
    for i, p in enumerate(patterns):
        noisy = p.copy()
        noisy[0] *= -1  # flip one bit
        recalled = net.recall(noisy)
        match = "✓" if np.array_equal(recalled, p.astype(np.int_)) else "✗"
        print(f"Pattern {i}: {p.astype(int)} → noisy: {noisy.astype(int)} → recalled: {recalled} {match}")

    print(f"\nRecall accuracy (10% noise): {net.test_recall(patterns, 0.1):.0%}")