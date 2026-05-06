"""Fuzzy set operations, relations, and max-min composition."""

import numpy as np


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


def fuzzy_relation(A, B):
    """Cartesian product relation: R[i][j] = min(A[i], B[j])."""
    return np.minimum(A[:, np.newaxis], B[np.newaxis, :])


def maxmin_composition(R1, R2):
    """Max-min composition of two fuzzy relations R1 (m×n) and R2 (n×p)."""
    if R1.shape[1] != R2.shape[0]:
        raise ValueError(f"Incompatible shapes: {R1.shape} and {R2.shape}")
    # R1[:, :, None] broadcasts over p; R2[None, :, :] broadcasts over m
    return np.max(np.minimum(R1[:, :, None], R2[None, :, :]), axis=1)


if __name__ == "__main__":
    A = np.array([0.2, 0.5, 0.7, 1.0])
    B = np.array([0.3, 0.6, 0.8, 0.4])

    print(f"A: {A}\nB: {B}\n")

    print("--- Fuzzy Set Operations ---")
    print(f"Union:         {fuzzy_union(A, B)}")
    print(f"Intersection:  {fuzzy_intersection(A, B)}")
    print(f"Complement(A): {fuzzy_complement(A)}")
    print(f"Difference:    {fuzzy_difference(A, B)}")
    print(f"Alpha-cut(A, 0.5): {fuzzy_alpha_cut(A, 0.5)}\n")

    R1 = fuzzy_relation(A, B)
    R2 = fuzzy_relation(B, A)
    print("--- Fuzzy Relations ---")
    print(f"R1 (A × B):\n{R1}\n")
    print(f"R2 (B × A):\n{R2}\n")

    print("--- Max-Min Composition ---")
    print(f"R1 ∘ R2:\n{maxmin_composition(R1, R2)}")
