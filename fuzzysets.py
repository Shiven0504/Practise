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


