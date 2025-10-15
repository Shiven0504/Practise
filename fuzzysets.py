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

# Specific Rotation of Cane Sugar Solution using Half-Shade Polarimeter

def specific_rotation(alpha, l_cm, weight_g, volume_ml):
    """
    Calculate specific rotation of cane sugar solution.

    Parameters:
    alpha    : observed rotation in degrees
    l_cm     : length of polarimeter tube in cm
    weight_g : weight of sugar dissolved (grams)
    volume_ml: volume of solution prepared (mL)

    Returns:
    Specific rotation [α]
    """
    # Convert path length to dm
    l_dm = l_cm / 10.0
    
    # Concentration (g per 100 mL)
    p = (weight_g / volume_ml) * 100
    
    # Formula: [α] = 100 * α / (l * p)
    specific_alpha = (100 * alpha) / (l_dm * p)
    return specific_alpha



# Example experiment calculation

# Suppose:
# - Observed rotation α = 13.2°
# - Tube length = 20 cm
# - Solution prepared: 2.0 g sucrose in 100 mL

alpha_obs = 13.2      # degrees
tube_length = 20.0    # cm
weight = 2.0          # g
volume = 100.0        # mL

result = specific_rotation(alpha_obs, tube_length, weight, volume)

print("Observed rotation (α):", alpha_obs, "degrees")
print("Tube length (l):", tube_length, "cm")
print("Concentration (p):", (weight/volume)*100, "g/100 mL")
print("Specific rotation [α]: {:.2f} ° dm⁻¹ (g/100mL)⁻¹".format(result))

