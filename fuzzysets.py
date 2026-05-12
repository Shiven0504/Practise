"""Fuzzy set operations, relations, and max-min composition."""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def fuzzy_union(A: NDArray, B: NDArray) -> NDArray:
    """
    Compute the union of two fuzzy sets using max operator.
    
    Args:
        A: First fuzzy set (array-like, values in [0, 1])
        B: Second fuzzy set (array-like, values in [0, 1])
    
    Returns:
        Union of A and B: element-wise maximum
    """
    return np.maximum(A, B)


def fuzzy_intersection(A: NDArray, B: NDArray) -> NDArray:
    """
    Compute the intersection of two fuzzy sets using min operator.
    
    Args:
        A: First fuzzy set (array-like, values in [0, 1])
        B: Second fuzzy set (array-like, values in [0, 1])
    
    Returns:
        Intersection of A and B: element-wise minimum
    """
    return np.minimum(A, B)


def fuzzy_complement(A: NDArray) -> NDArray:
    """
    Compute the complement of a fuzzy set.
    
    Args:
        A: Fuzzy set (array-like, values in [0, 1])
    
    Returns:
        Complement of A: 1 - A
    """
    return 1 - A


def fuzzy_difference(A: NDArray, B: NDArray) -> NDArray:
    """
    Compute the difference of two fuzzy sets: A - B = A ∩ B^c.
    
    Args:
        A: First fuzzy set (array-like, values in [0, 1])
        B: Second fuzzy set (array-like, values in [0, 1])
    
    Returns:
        Difference: element-wise min(A, 1 - B)
    """
    return np.minimum(A, 1 - B)


def fuzzy_relation(A: NDArray, B: NDArray) -> NDArray:
    """
    Compute the Cartesian product relation of two fuzzy sets.
    
    Args:
        A: First fuzzy set (1D array, values in [0, 1])
        B: Second fuzzy set (1D array, values in [0, 1])
    
    Returns:
        Fuzzy relation R where R[i][j] = min(A[i], B[j]) (2D array)
    """
    return np.minimum(A[:, np.newaxis], B[np.newaxis, :])


def fuzzy_alpha_cut(A: NDArray, alpha: float) -> NDArray:
    """
    Compute the alpha-cut of a fuzzy set (crisp set).
    
    Args:
        A: Fuzzy set (array-like, values in [0, 1])
        alpha: Threshold value in [0, 1]
    
    Returns:
        Crisp set with 1 where A >= alpha, 0 elsewhere
    """
    return np.where(A >= alpha, 1, 0)


def maxmin_composition(R1: NDArray, R2: NDArray) -> NDArray:
    """
    Compute the max-min composition of two fuzzy relations.
    
    Args:
        R1: First fuzzy relation (m × n matrix)
        R2: Second fuzzy relation (n × p matrix)
    
    Returns:
        Composed relation (m × p matrix) where each element is
        max over all k of min(R1[i][k], R2[k][j])
    
    Raises:
        ValueError: If R1.shape[1] != R2.shape[0]
    """
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
