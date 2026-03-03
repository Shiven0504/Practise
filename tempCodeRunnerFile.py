"""
Hopfield Network — Bipolar Auto-Associative Memory

Stores patterns via Hebbian learning (outer product rule) and recalls
them from noisy/partial inputs using synchronous iterative updates.
"""

import numpy as np


# ── Stored patterns ──────────────────────────────────────────────────
A1 = np.array([-1,  1, -1,  1])
A2 = np.array([ 1,  1,  1, -1])
A3 = np.array([-1, -1, -1,  1])
STORED_PATTERNS = [A1, A2, A3]
PATTERN_NAMES   = ["A1", "A2", "A3"]


# ── Weight matrix (Hebbian rule, zero diagonal) ─────────────────────
def build_weight_matrix(patterns: list[np.ndarray]) -> np.ndarray:
    n = patterns[0].shape[0]
    W = sum(np.outer(p, p) for p in patterns)
    np.fill_diagonal(W, 0)
    return W


# ── Activation ───────────────────────────────────────────────────────
def bipolar_activation(x: np.ndarray) -> np.ndarray:
    """Bipolar step: x >= 0 → +1, else → -1."""
    return np.where(x >= 0, 1, -1)


# ── Energy ───────────────────────────────────────────────────────────
def energy(state: np.ndarray, W: np.ndarray) -> float:
    """Hopfield energy: E = -0.5 * s^T W s"""
    return -0.5 * state @ W @ state


# ── Hamming distance ─────────────────────────────────────────────────
def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


# ── Recall ───────────────────────────────────────────────────────────
def recall(
    pattern: np.ndarray,
    W: np.ndarray,
    activation_fn=bipolar_activation,
    max_steps: int = 20,
) -> tuple[np.ndarray, int, bool, list[float]]:
    """
    Synchronous iterative recall.

    Returns:
        final_state, steps_taken, converged, energy_trace
    """
    state = pattern.copy()
    energy_trace = [energy(state, W)]

    for step in range(1, max_steps + 1):
        new_state = activation_fn(W @ state)
        energy_trace.append(energy(new_state, W))
        if np.array_equal(new_state, state):
            return new_state, step, True, energy_trace
        state = new_state

    return state, max_steps, False, energy_trace


# ── Pattern matching ─────────────────────────────────────────────────
def match_stored(
    state: np.ndarray,
    stored: list[np.ndarray],
    names: list[str] | None = None,
) -> tuple[str | None, int]:
    """Return (name, index) of matching stored pattern, or (None, -1)."""
    for i, p in enumerate(stored):
        if np.array_equal(state, p):
            return (names[i] if names else str(i), i)
    return (None, -1)


# ── Test harness ─────────────────────────────────────────────────────
TEST_PATTERNS = {
    "Ax": np.array([-1,  1, -1,  1]),   # identical to A1
    "Ay": np.array([ 1,  1,  1,  1]),   # 1-bit flip from A2
    "Az": np.array([-1, -1, -1, -1]),   # 1-bit flip from A3
}


def main() -> None:
    W = build_weight_matrix(STORED_PATTERNS)

    print("Weight Matrix (W):")
    print(W)
    print(f"\nStored {len(STORED_PATTERNS)} patterns of dimension {STORED_PATTERNS[0].shape[0]}")
    print(f"Theoretical capacity ≈ {int(STORED_PATTERNS[0].shape[0] / (2 * np.log2(STORED_PATTERNS[0].shape[0])))} patterns\n")

    print("─" * 50)
    print("Recall Tests (synchronous update)")
    print("─" * 50)

    for name, pattern in TEST_PATTERNS.items():
        final, steps, converged, e_trace = recall(pattern, W, max_steps=20)
        match_name, idx = match_stored(final, STORED_PATTERNS, PATTERN_NAMES)

        print(f"\n  {name} = {pattern}")
        print(f"  Output  = {final}  (after {steps} step{'s' if steps != 1 else ''})")
        print(f"  Energy  : {e_trace[0]:.1f} → {e_trace[-1]:.1f}")
        print(f"  Hamming : {hamming_distance(pattern, final)} bit(s) changed")

        if converged and match_name:
            print(f"  Result  : ✓ converged → {match_name}")
        elif converged:
            print(f"  Result  : converged → spurious attractor")
        else:
            print(f"  Result  : ✗ did not converge in {steps} steps")


if __name__ == "__main__":
    main()