"""

import numpy as np

# Fuzzy Set Operations

def fuzzy_union(A, B):
    return np.maximum(A, B)

def fuzzy_intersection(A, B):
    return np.minimum(A, B)

def fuzzy_complement(A):
    return 1 - A

def fuzzy_difference(A, B):
    return np.minimum(A, fuzzy_complement(B))



# Fuzzy Relation (Cartesian Product)

def fuzzy_relation(A, B):
    # Cartesian product: min(A(x), B(y)) for each pair (x, y)
    relation = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            relation[i][j] = min(A[i], B[j])
    return relation


# Max–Min Composition

def maxmin_composition(R1, R2):
    # R1: m x n, R2: n x p
    m, n = R1.shape
    n2, p = R2.shape
    if n != n2:
        raise ValueError("Incompatible relation sizes for composition")

    R = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            R[i][j] = np.max(np.minimum(R1[i, :], R2[:, j]))
    return R


# Example Usage

# Define two fuzzy sets A and B
A = np.array([0.2, 0.5, 0.7, 1.0])   # fuzzy set A
B = np.array([0.3, 0.6, 0.8, 0.4])   # fuzzy set B

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)

# Perform basic fuzzy set operations
print("\n--- Fuzzy Set Operations ---")
print("Union:", fuzzy_union(A, B))
print("Intersection:", fuzzy_intersection(A, B))
print("Complement of A:", fuzzy_complement(A))
print("Difference (A - B):", fuzzy_difference(A, B))

# Create fuzzy relation using Cartesian Product
print("\n--- Fuzzy Relation ---")
R1 = fuzzy_relation(A, B)
print("Relation R1 (A x B):\n", R1)

# Another fuzzy relation (B x A for example)
R2 = fuzzy_relation(B, A)
print("Relation R2 (B x A):\n", R2)

# Max-Min Composition of R1 and R2
print("\n--- Max-Min Composition ---")
R_comp = maxmin_composition(R1, R2)
print("R1 o R2:\n", R_comp)

"""

# ...existing code...
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


def match_stored(state, stored, names=None):
    """Return name/index of matching stored pattern or (None, -1) if no match."""
    for i, p in enumerate(stored):
        if np.array_equal(state, p):
            return (names[i] if names else i, i)
    return (None, -1)


# test patterns
Ax = np.array([-1,  1, -1,  1])
Ay = np.array([ 1,  1,  1,  1])
Az = np.array([-1, -1, -1, -1])
test_patterns = [Ax, Ay, Az]
test_names = ["Ax", "Ay", "Az"]

if __name__ == "__main__":
    print("\n--- Testing Network Recall (iterative) ---")
    for name, pattern in zip(test_names, test_patterns):
        print(f"\nTesting with pattern: {name}")
        print("Input:")
        print(pattern)

        final, steps, converged = recall(pattern, W, activation, max_steps=20)

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