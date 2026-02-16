# ...existing code...
"""
Improved fuzzy set utilities:
- input validation and clamping to [0,1]
- vectorized numpy operations for performance
- clearer docstrings and example guarded by __main__
- optimized max-min composition using broadcasting
"""

import numpy as np
from typing import Iterable

def _as_fuzzy_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("fuzzy sets must be 1-D arrays or sequences")
    if not np.isfinite(a).all():
        raise ValueError("fuzzy set must contain finite numbers")
    # clamp values into [0,1]
    a = np.clip(a, 0.0, 1.0)
    return a

def fuzzy_union(A: Iterable, B: Iterable) -> np.ndarray:
    """Elementwise max (union)"""
    A, B = _as_fuzzy_array(A), _as_fuzzy_array(B)
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    return np.maximum(A, B)

def fuzzy_intersection(A: Iterable, B: Iterable) -> np.ndarray:
    """Elementwise min (intersection)"""
    A, B = _as_fuzzy_array(A), _as_fuzzy_array(B)
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    return np.minimum(A, B)

def fuzzy_complement(A: Iterable) -> np.ndarray:
    """Complement: 1 - membership"""
    A = _as_fuzzy_array(A)
    return 1.0 - A

def fuzzy_difference(A: Iterable, B: Iterable) -> np.ndarray:
    """A \ B = A ∧ (¬B)"""
    A, B = _as_fuzzy_array(A), _as_fuzzy_array(B)
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    return np.minimum(A, fuzzy_complement(B))


def fuzzy_relation(A: Iterable, B: Iterable) -> np.ndarray:
    """
    Cartesian fuzzy relation R(x,y) = min(A(x), B(y))
    Returns an array of shape (len(A), len(B))
    """
    A = _as_fuzzy_array(A)
    B = _as_fuzzy_array(B)
    # vectorized outer minimum
    return np.minimum(A[:, None], B[None, :])


def maxmin_composition(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """
    Max-Min composition R = R1 o R2 where
      R1 is m x n, R2 is n x p
    Returns m x p matrix with R[i,j] = max_k min(R1[i,k], R2[k,j])
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    if R1.ndim != 2 or R2.ndim != 2:
        raise ValueError("R1 and R2 must be 2-D arrays")
    m, n = R1.shape
    n2, p = R2.shape
    if n != n2:
        raise ValueError("Incompatible shapes for composition: R1.shape[1] must equal R2.shape[0]")
    # clamp into [0,1] to avoid invalid memberships
    R1 = np.clip(R1, 0.0, 1.0)
    R2 = np.clip(R2, 0.0, 1.0)
    # mins has shape (m, n, p); take max over axis=1 (k)
    mins = np.minimum(R1[:, :, None], R2[None, :, :])
    return np.max(mins, axis=1)


if __name__ == "__main__":
    # simple demonstration / smoke test
    A = np.array([0.2, 0.5, 0.7, 1.0])
    B = np.array([0.3, 0.6, 0.8, 0.4])

    print("Fuzzy Set A:", A)
    print("Fuzzy Set B:", B)

    print("\n--- Fuzzy Set Operations ---")
    print("Union:", fuzzy_union(A, B))
    print("Intersection:", fuzzy_intersection(A, B))
    print("Complement of A:", fuzzy_complement(A))
    print("Difference (A - B):", fuzzy_difference(A, B))

    print("\n--- Fuzzy Relation ---")
    R1 = fuzzy_relation(A, B)
    print("Relation R1 (A x B):\n", R1)

    R2 = fuzzy_relation(B, A)
    print("Relation R2 (B x A):\n", R2)

    print("\n--- Max-Min Composition ---")
    R_comp = maxmin_composition(R1, R2)
    print("R1 o R2:\n", R_comp)