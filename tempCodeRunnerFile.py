import numpy as np

A1 = np.array([-1,  1, -1,  1])
A2 = np.array([ 1,  1,  1, -1])
A3 = np.array([-1, -1, -1,  1])
stored_patterns = [A1, A2, A3]
pattern_names = ["A1", "A2", "A3"]

W = np.zeros((4, 4))
for A in stored_patterns:
    W += np.outer(A, A)

np.fill_diagonal(W, 0)

print("Weight Matrix (W):")
print(W)


def activation(x):
    """Bipolar step activation function."""
    return np.where(x >= 0, 1, -1)


def recall(pattern, W, activation_fn, max_steps=10):
    """
    Synchronous iterative recall: apply activation(np.dot(state, W)) until convergence
    or max_steps reached. Returns (final_state, steps_taken, converged_bool).
    """
    state = pattern.copy()
    for step in range(1, max_steps + 1):
        new_state = activation_fn(np.dot(state, W))
        if np.array_equal(new_state, state):
            return new_state, step, True
        state = new_state
    return state, max_steps, False


def energy(state, W):
    """Compute Hopfield network energy for a bipolar state."""
    return -0.5 * float(state @ W @ state)


def match_stored(state, stored, names=None):
    """Return name/index of matching stored pattern or (None, -1) if no match."""
    for i, p in enumerate(stored):
        if np.array_equal(state, p):
            return (names[i] if names else i, i)
    return (None, -1)


def flip_bits(pattern, n_bits=1, seed=None):
    """Return a copy of pattern with n_bits randomly flipped."""
    rng = np.random.default_rng(seed)
    out = pattern.copy()
    idx = rng.choice(pattern.size, size=min(n_bits, pattern.size), replace=False)
    out[idx] = -out[idx]
    return out


# test patterns (clean + noisy)
Ax = np.array([-1,  1, -1,  1])
Ay = np.array([ 1,  1,  1,  1])
Az = np.array([-1, -1, -1, -1])
Ax_noisy = flip_bits(Ax, n_bits=1, seed=0)
test_patterns = [Ax, Ax_noisy, Ay, Az]
test_names = ["Ax", "Ax_noisy(1bit)", "Ay", "Az"]

if __name__ == "__main__":
    print("\n--- Testing Network Recall (iterative) ---")
    for name, pattern in zip(test_names, test_patterns):
        print(f"\nTesting with pattern: {name}")
        print(f"  Input:  {pattern}")
        print(f"  Energy (input): {energy(pattern, W):.2f}")

        final, steps, converged = recall(pattern, W, activation, max_steps=20)
        match_name, _ = match_stored(final, stored_patterns, pattern_names)

        print(f"  Output: {final}  (after {steps} step(s), converged={converged})")
        print(f"  Energy (output): {energy(final, W):.2f}")
        print(f"  Match: {match_name or 'no stored pattern'}")
