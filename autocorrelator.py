# ...existing code...
"""
Hopfield-style associative memory demo (autocorrelator).

Improvements:
- clearer docstrings and small refactors
- more robust activation (accepts scalars/arrays)
- minor safety checks and deterministic RNG handling
- improved matching return semantics
- cleaner output formatting
"""
import numpy as np
from typing import Tuple, List, Optional

A1 = np.array([-1,  1, -1,  1])
A2 = np.array([ 1,  1,  1, -1])
A3 = np.array([-1, -1, -1,  1])
stored_patterns = [A1, A2, A3]
pattern_names = ["A1", "A2", "A3"]


def build_weight_matrix(patterns: List[np.ndarray]) -> np.ndarray:
    """Build a symmetric Hopfield weight matrix from bipolar patterns (zero diagonal)."""
    if not patterns:
        raise ValueError("patterns list must not be empty")
    n = patterns[0].size
    W = np.zeros((n, n), dtype=float)
    for p in patterns:
        if p.size != n:
            raise ValueError("all patterns must have the same length")
        W += np.outer(p, p)
    np.fill_diagonal(W, 0.0)
    return W


def activation(x: np.ndarray) -> np.ndarray:
    """Bipolar step activation. Accepts scalars or arrays and returns same-shape output of -1 or 1."""
    xa = np.asarray(x)
    return np.where(xa >= 0, 1, -1)


def energy(state: np.ndarray, W: np.ndarray) -> float:
    """Compute Hopfield network energy for a bipolar state."""
    s = np.asarray(state, dtype=float)
    return float(-0.5 * (s @ W @ s))


def recall(pattern: np.ndarray,
           W: np.ndarray,
           activation_fn=activation,
           max_steps: int = 10,
           mode: str = "synchronous",
           rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, int, bool]:
    """
    Recall a pattern using the weight matrix W.
    mode: 'synchronous' or 'asynchronous'.
    Returns: (final_state, steps_taken, converged_bool).
    """
    if rng is None:
        rng = np.random.default_rng()

    state = pattern.copy()
    n = state.size

    if mode == "synchronous":
        for step in range(1, max_steps + 1):
            # synchronous update: compute net input for all neurons
            new_state = activation_fn(W @ state)
            if np.array_equal(new_state, state):
                return new_state, step, True
            state = new_state
        return state, max_steps, False

    elif mode == "asynchronous":
        for step in range(1, max_steps + 1):
            prev = state.copy()
            # one full random-order pass over neurons is counted as one step
            for i in rng.permutation(n):
                net = float(W[i, :] @ state)
                # activation_fn may return scalar or 0-d array; ensure scalar int assignment
                state[i] = int(activation_fn(net))
            if np.array_equal(state, prev):
                return state, step, True
        return state, max_steps, False

    else:
        raise ValueError("mode must be 'synchronous' or 'asynchronous'")


def match_stored(state: np.ndarray, stored: List[np.ndarray], names: Optional[List[str]] = None) -> Tuple[Optional[str], int]:
    """
    Return (name_or_None, index) of matching stored pattern, or (None, -1) if no match.
    If names is provided it will be used for the returned name.
    """
    for i, p in enumerate(stored):
        if np.array_equal(state, p):
            return (names[i] if names is not None else None, i)
    return (None, -1)


def flip_bits(pattern: np.ndarray, n_bits: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Return a copy of pattern with n_bits randomly flipped (bipolar)."""
    rng = np.random.default_rng(seed)
    out = pattern.copy()
    idx = rng.choice(pattern.size, size=min(max(0, n_bits), pattern.size), replace=False)
    out[idx] = -out[idx]
    return out


def _format_result(mode: str, steps: int, final: np.ndarray, converged: bool, W: np.ndarray, stored: List[np.ndarray], names: List[str]) -> str:
    match_name, _ = match_stored(final, stored, names)
    return f"  [{mode:11s}] Output after {steps} step(s): {final}  Converged: {converged}  Match: {match_name}\n              Energy (output): {energy(final, W):.3f}"


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
        print(f"Energy (input): {energy(pattern, W):.3f}")

        for mode in ("synchronous", "asynchronous"):
            final, steps, converged = recall(pattern, W, max_steps=20, mode=mode, rng=rng)
            print(_format_result(mode, steps, final, converged, W, stored_patterns, pattern_names))
# ...existing code...