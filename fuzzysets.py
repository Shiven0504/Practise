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

// ...existing code...
# --- 2) Fuzzy Logic (using scikit-fuzzy) ---
def fuzzy_temperature_demo(sample_temps=None, plot=True, save_plot=False, out_path="temperature_fuzzy_sets.png"):
    """
    Demonstrate temperature fuzzy sets (cold, warm, hot), print a table of memberships
    for sample temperatures and optionally plot the membership functions.
    """
    import skfuzzy as fuzz
    import numpy as np
    import matplotlib.pyplot as plt

    x_temp = np.arange(0, 41, 1)                 # universe: 0..40 °C
    cold = fuzz.trimf(x_temp, [0, 0, 20])
    warm = fuzz.trimf(x_temp, [10, 20, 30])
    hot  = fuzz.trimf(x_temp, [20, 40, 40])

    if sample_temps is None:
        sample_temps = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])

    cold_m = np.array([fuzz.interp_membership(x_temp, cold, t) for t in sample_temps])
    warm_m = np.array([fuzz.interp_membership(x_temp, warm, t) for t in sample_temps])
    hot_m  = np.array([fuzz.interp_membership(x_temp, hot, t)  for t in sample_temps])

    # Print a tidy table
    print("\n--- Fuzzy Logic Example (Temperature) ---")
    print("Temp |  cold  |  warm  |   hot ")
    print("--------------------------------")
    for t, c, w, h in zip(sample_temps, cold_m, warm_m, hot_m):
        print(f"{t:4d} | {c:6.3f} | {w:6.3f} | {h:6.3f}")

    if plot:
        plt.figure(figsize=(7, 3.5))
        plt.plot(x_temp, cold, label="cold", lw=2)
        plt.plot(x_temp, warm, label="warm", lw=2)
        plt.plot(x_temp, hot,  label="hot",  lw=2)
        plt.scatter(sample_temps, cold_m, c='C0', s=25, label=None, zorder=5)
        plt.scatter(sample_temps, warm_m, c='C1', s=25, label=None, zorder=5)
        plt.scatter(sample_temps, hot_m,  c='C2', s=25, label=None, zorder=5)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Membership degree")
        plt.title("Temperature fuzzy sets")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        if save_plot:
            plt.savefig(out_path, dpi=150)
            print(f"Saved plot to: {out_path}")
        plt.show()

# Call demo with defaults
fuzzy_temperature_demo()
