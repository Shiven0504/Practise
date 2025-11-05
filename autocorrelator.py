import numpy as np

A1 = np.array([-1,  1, -1,  1])
A2 = np.array([ 1,  1,  1, -1])
A3 = np.array([-1, -1, -1,  1])
stored_patterns = [A1, A2, A3]
pattern_names = ["A1", "A2", "A3"]


def build_weight_matrix(patterns):
    """Builds a Hopfield-style weight matrix from bipolar patterns (zero diagonal)."""
    n = patterns[0].size
    W = np.zeros((n, n), dtype=float)
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W


def activation(x):
    """Bipolar step activation function."""
    return np.where(x >= 0, 1, -1)


def recall(pattern, W, activation_fn=activation, max_steps=10):
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


def match_stored(state, stored, names=None):
    """Return (name, index) of matching stored pattern or (None, -1)."""
    for i, p in enumerate(stored):
        if np.array_equal(state, p):
            return (names[i] if names else i, i)
    return (None, -1)


# build weight matrix once
W = build_weight_matrix(stored_patterns)

if __name__ == "__main__":
    print("Weight Matrix (W):")
    print(W)

    # test inputs
    Ax = np.array([-1,  1, -1,  1])
    Ay = np.array([ 1,  1,  1,  1])
    Az = np.array([-1, -1, -1, -1])
    test_patterns = [Ax, Ay, Az]
    test_names = ["Ax", "Ay", "Az"]

    print("\n--- Testing Network Recall (iterative) ---")
    for name, pattern in zip(test_names, test_patterns):
        print(f"\nTesting with pattern: {name}")
        print("Input:")
        print(pattern)

        final, steps, converged = recall(pattern, W, max_steps=20)

        print(f"Output after {steps} step(s):")
        print(final)
        if converged:
            match_name, idx = match_stored(final, stored_patterns, pattern_names)
            if match_name is not None:
                print(f"Result: Converged to stored pattern {match_name} (index {idx})")
            else:
                print("Result: Converged to a pattern but not one of the stored patterns")
        else:
            print("Result: Did not converge within max steps")