"""Hopfield associative memory recall demo.

Builds a weight matrix from bipolar stored patterns and
runs synchronous recall tests for exact and noisy inputs.
"""

import numpy as np


def build_weight_matrix(patterns):
    """Build a Hopfield weight matrix for a list of bipolar patterns."""
    size = len(patterns[0])
    W = np.zeros((size, size), dtype=int)
    for pattern in patterns:
        W += np.outer(pattern, pattern)
    np.fill_diagonal(W, 0)
    return W


def bipolar_activation(x):
    """Return bipolar activation values from input potentials.

    Values greater than or equal to zero map to +1, otherwise -1.
    """
    return np.where(x >= 0, 1, -1)


def recall(pattern, W, activation_fn=bipolar_activation, max_steps=10):
    """Run synchronous recall until convergence or max_steps.

    Returns (final_state, steps_taken, converged_bool).
    """
    state = pattern.copy()
    for step in range(1, max_steps + 1):
        new_state = activation_fn(np.dot(state, W))
        if np.array_equal(new_state, state):
            return new_state, step, True
        state = new_state
    return state, max_steps, False


def match_stored(state, stored, names=None):
    """Return the stored pattern name/index that matches the recalled state."""
    for i, pattern in enumerate(stored):
        if np.array_equal(state, pattern):
            return (names[i] if names else i, i)
    return (None, -1)


def flip_bits(pattern, num_flips, seed=None):
    """Return a noisy version of the pattern with num_flips bipolar bit flips."""
    rng = np.random.default_rng(seed)
    noisy = pattern.copy()
    if num_flips > 0:
        indices = rng.choice(len(pattern), size=num_flips, replace=False)
        noisy[indices] *= -1
    return noisy


def describe_pattern(name, pattern):
    print(f"{name}: {pattern}")


def demo_recall(name, pattern, W, stored_patterns, pattern_names):
    print(f"\nTesting with pattern: {name}")
    describe_pattern("Input", pattern)
    final, steps, converged = recall(pattern, W, max_steps=20)
    describe_pattern("Output", final)
    print(f"Steps: {steps}")
    if converged:
        match_name, idx = match_stored(final, stored_patterns, pattern_names)
        if match_name is not None:
            print(f"Result: Converged to stored pattern {match_name} (index {idx})")
        else:
            print("Result: Converged to a pattern but not one of the stored patterns")
    else:
        print("Result: Did not converge within max steps")


def demo_noisy_recall(stored_patterns, pattern_names, W):
    print("\n--- Noisy recall demo for stored patterns ---")
    for name, pattern in zip(pattern_names, stored_patterns):
        for flips in (0, 1, 2):
            noisy_pattern = flip_bits(pattern, flips, seed=42)
            print(f"\n{name} with {flips} flip(s):")
            describe_pattern("Noisy input", noisy_pattern)
            final, steps, _ = recall(noisy_pattern, W, max_steps=20)
            describe_pattern("Output", final)
            match_name, idx = match_stored(final, stored_patterns, pattern_names)
            if match_name is not None:
                print(f"Result: Converged to stored pattern {match_name} (index {idx})")
            else:
                print("Result: Did not recall a stored pattern")


A1 = np.array([-1,  1, -1,  1])
A2 = np.array([ 1,  1,  1, -1])
A3 = np.array([-1, -1, -1,  1])
stored_patterns = [A1, A2, A3]
pattern_names = ["A1", "A2", "A3"]

Ax = np.array([-1,  1, -1,  1])
Ay = np.array([ 1,  1,  1,  1])
Az = np.array([-1, -1, -1, -1])
test_patterns = [Ax, Ay, Az]
test_names = ["Ax", "Ay", "Az"]


def main():
    W = build_weight_matrix(stored_patterns)
    print("Weight Matrix (W):")
    print(W)
    print("\n--- Testing Network Recall (iterative) ---")
    for name, pattern in zip(test_names, test_patterns):
        demo_recall(name, pattern, W, stored_patterns, pattern_names)
    demo_noisy_recall(stored_patterns, pattern_names, W)


if __name__ == "__main__":
    main()



