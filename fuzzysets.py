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


def specific_rotation(alpha: float, l_cm: float, weight_g: float, volume_ml: float) -> float:
    """
    Calculate specific rotation [α] of a solution.

    Parameters
    ----------
    alpha : float
        Observed rotation in degrees.
    l_cm : float
        Polarimeter tube length in centimeters.
    weight_g : float
        Mass of solute (grams) dissolved in the solution.
    volume_ml : float
        Volume of the prepared solution in millilitres.

    Returns
    -------
    float
        Specific rotation [α] in degrees · dm⁻¹ · (g/100mL)⁻¹

    Raises
    ------
    ValueError
        If l_cm <= 0, volume_ml <= 0, or weight_g < 0.
    """
    # Basic validation
    if l_cm <= 0:
        raise ValueError("Tube length (l_cm) must be > 0 cm")
    if volume_ml <= 0:
        raise ValueError("Volume (volume_ml) must be > 0 mL")
    if weight_g < 0:
        raise ValueError("Weight (weight_g) must be >= 0 g")

    # Convert path length to dm
    l_dm = l_cm / 10.0

    # Concentration in g per 100 mL
    p = (weight_g / volume_ml) * 100.0

    if p == 0:
        raise ValueError("Concentration is zero (no solute); specific rotation undefined")

    # Formula: [α] = 100 * α / (l * p)
    specific_alpha = (100.0 * alpha) / (l_dm * p)
    return specific_alpha


if __name__ == "__main__":
    # Improved CLI-driven example + nicer output formatting
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="specific_rotation",
        description="Compute specific rotation [α] for a solution (° · dm⁻¹ · (g/100mL)⁻¹)."
    )
    parser.add_argument("--alpha", type=float, default=13.2, help="Observed rotation in degrees (default: 13.2)")
    parser.add_argument("--length-cm", type=float, dest="l_cm", default=20.0, help="Tube length in cm (default: 20.0)")
    parser.add_argument("--weight-g", type=float, dest="weight_g", default=2.0, help="Mass of solute in g (default: 2.0)")
    parser.add_argument("--volume-ml", type=float, dest="volume_ml", default=100.0, help="Volume of solution in mL (default: 100.0)")
    args = parser.parse_args()

    try:
        result = specific_rotation(args.alpha, args.l_cm, args.weight_g, args.volume_ml)
    except ValueError as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)
    else:
        conc = (args.weight_g / args.volume_ml) * 100.0
        print("Specific Rotation Calculation")
        print("-----------------------------")
        print(f"Observed rotation (α): {args.alpha:.3f} °")
        print(f"Tube length (l):      {args.l_cm:.2f} cm")
        print(f"Mass of solute:       {args.weight_g:.3f} g")
        print(f"Volume of solution:    {args.volume_ml:.2f} mL")
        print(f"Concentration (p):    {conc:.3f} g/100mL")
        print(f"\nSpecific Rotation [α]: {result:.6f} ° · dm⁻¹ · (g/100mL)⁻¹")