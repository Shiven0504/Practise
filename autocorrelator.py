import numpy as np
from typing import Tuple, List, Optional

A1 = np.array([-1,  1, -1,  1])
A2 = np.array([ 1,  1,  1, -1])
A3 = np.array([-1, -1, -1,  1])
stored_patterns = [A1, A2, A3]
pattern_names = ["A1", "A2", "A3"]


def build_weight_matrix(patterns: List[np.ndarray]) -> np.ndarray:
    """Builds a Hopfield-style weight matrix from bipolar patterns (zero diagonal)."""
    n = patterns[0].size
    W = np.zeros((n, n), dtype=float)
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W


def activation(x: np.ndarray) -> np.ndarray:
    """Bipolar step activation function."""
    return np.where(x >= 0, 1, -1)


def energy(state: np.ndarray, W: np.ndarray) -> float:
    """Compute Hopfield network energy for a bipolar state."""
    return -0.5 * float(state @ W @ state)


def recall(pattern: np.ndarray,
           W: np.ndarray,
           activation_fn=activation,
           max_steps: int = 10,
           mode: str = "synchronous",
           rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, int, bool]:
    """
    Recall a pattern. mode: 'synchronous' (default) or 'asynchronous'.
    Returns (final_state, steps_taken, converged_bool).
    """
    if rng is None:
        rng = np.random.default_rng()

    state = pattern.copy()
    n = state.size

    if mode == "synchronous":
        for step in range(1, max_steps + 1):
            new_state = activation_fn(np.dot(state, W))
            if np.array_equal(new_state, state):
                return new_state, step, True
            state = new_state
        return state, max_steps, False

    elif mode == "asynchronous":
        for step in range(1, max_steps + 1):
            prev = state.copy()
            # random order asynchronous updates (one full pass = one step)
            for i in rng.permutation(n):
                net = W[i, :] @ state
                state[i] = activation_fn(np.array([net]))[0]
            if np.array_equal(state, prev):
                return state, step, True
        return state, max_steps, False

    else:
        raise ValueError("mode must be 'synchronous' or 'asynchronous'")


def match_stored(state: np.ndarray, stored: List[np.ndarray], names: Optional[List[str]] = None) -> Tuple[Optional[str], int]:
    """Return (name, index) of matching stored pattern or (None, -1)."""
    for i, p in enumerate(stored):
        if np.array_equal(state, p):
            return (names[i] if names else i, i)
    return (None, -1)


def flip_bits(pattern: np.ndarray, n_bits: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Return a copy of pattern with n_bits randomly flipped (bipolar)."""
    rng = np.random.default_rng(seed)
    out = pattern.copy()
    idx = rng.choice(pattern.size, size=min(n_bits, pattern.size), replace=False)
    out[idx] = -out[idx]
    return out


# build weight matrix when module run (not on import)
if __name__ == "__main__":
    W = build_weight_matrix(stored_patterns)

    print("Weight Matrix (W):")
    print(W)

    # test inputs (including a noisy version)
    Ax = np.array([-1,  1, -1,  1])
    Ay = np.array([ 1,  1,  1,  1])
    Az = np.array([-1, -1, -1, -1])
    Ax_noisy = flip_bits(Ax, n_bits=1, seed=0)

    test_patterns = [Ax, Ax_noisy, Ay, Az]
    test_names = ["Ax", "Ax_noisy(1bit)", "Ay", "Az"]

    rng = np.random.default_rng(0)

    print("\n--- Testing Network Recall ---")
    for name, pattern in zip(test_names, test_patterns):
        print(f"\nTesting with pattern: {name}")
        print("Input:", pattern)
        print("Energy (input):", energy(pattern, W))

        for mode in ("synchronous", "asynchronous"):
            final, steps, converged = recall(pattern, W, max_steps=20, mode=mode, rng=rng)
            match_name, idx = match_stored(final, stored_patterns, pattern_names)
            print(f"  [{mode:11s}] Output after {steps} step(s): {final}  Converged: {converged}  Match: {match_name}")
            print(f"              Energy (output): {energy(final, W)}")