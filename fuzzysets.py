"""Fuzzy set operations, relations, and max-min composition."""

import numpy as np

np.set_printoptions(precision=2)


# --- Core Operations ---

def fuzzy_union(A, B):
    """Union of two fuzzy sets: max(A, B)."""
    return np.maximum(A, B)


def fuzzy_intersection(A, B):
    """Intersection of two fuzzy sets: min(A, B)."""
    return np.minimum(A, B)


def fuzzy_complement(A):
    """Complement of a fuzzy set: 1 - A."""
    return 1 - A


def fuzzy_difference(A, B):
    """Difference of fuzzy sets: min(A, 1 - B)."""
    return np.minimum(A, 1 - B)


def algebraic_sum(A, B):
    """Algebraic sum (t-conorm): A + B - A*B."""
    return A + B - A * B


def algebraic_product(A, B):
    """Algebraic product (t-norm): A * B."""
    return A * B


# --- Relations ---

def fuzzy_relation(A, B):
    """Cartesian product relation: R[i][j] = min(A[i], B[j])."""
    return np.minimum(A[:, np.newaxis], B[np.newaxis, :])


def maxmin_composition(R1, R2):
    """Max-min composition of two fuzzy relations R1 (m×n) and R2 (n×p)."""
    if R1.shape[1] != R2.shape[0]:
        raise ValueError(
            f"Incompatible shapes for composition: {R1.shape} and {R2.shape}"
        )
    # Vectorized: broadcast R1 (m,n,1) and R2 (1,n,p), min along axis 1, max along axis 1
    return np.max(np.minimum(R1[:, :, np.newaxis], R2[np.newaxis, :, :]), axis=1)


# --- Verification ---

def verify_demorgan(A, B):
    """Verify De Morgan's Law: complement(A ∪ B) == complement(A) ∩ complement(B)."""
    lhs = fuzzy_complement(fuzzy_union(A, B))
    rhs = fuzzy_intersection(fuzzy_complement(A), fuzzy_complement(B))
    return np.allclose(lhs, rhs), lhs, rhs


if __name__ == "__main__":
    A = np.array([0.2, 0.5, 0.7, 1.0])
    B = np.array([0.3, 0.6, 0.8, 0.4])

    print(f"A: {A}")
    print(f"B: {B}\n")

    print("--- Fuzzy Set Operations ---")
    print(f"Union:            {fuzzy_union(A, B)}")
    print(f"Intersection:     {fuzzy_intersection(A, B)}")
    print(f"Complement(A):    {fuzzy_complement(A)}")
    print(f"Difference(A-B):  {fuzzy_difference(A, B)}")
    print(f"Algebraic Sum:    {algebraic_sum(A, B)}")
    print(f"Algebraic Product:{algebraic_product(A, B)}\n")

    print("--- De Morgan's Law Verification ---")
    holds, lhs, rhs = verify_demorgan(A, B)
    print(f"C(A ∪ B):     {lhs}")
    print(f"C(A) ∩ C(B):  {rhs}")
    print(f"Holds: {holds}\n")

    print("--- Fuzzy Relations ---")
    R1 = fuzzy_relation(A, B)
    R2 = fuzzy_relation(B, A)
    print(f"R1 (A × B):\n{R1}\n")
    print(f"R2 (B × A):\n{R2}\n")

    print("--- Max-Min Composition ---")
    print(f"R1 ∘ R2:\n{maxmin_composition(R1, R2)}")
