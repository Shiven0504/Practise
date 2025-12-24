
"""
from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np

#print(sns.get_dataset_names())
tips = sns.load_dataset("tips")
#print(tips.head())
iris = sns.load_dataset("iris")
#print(iris.head())
titanic = sns.load_dataset("titanic")
#print(titanic.head())
planets = sns.load_dataset("planets")
#print(planets.head())


#sns.scatterplot(x="tip", y="total_bill", data=tips, hue="day", size="size", palette="YlGnBu")

#sns.histplot(tips['tip'], kde=True, bins=30)

#sns.boxplot(x="day", y="tip", data=tips, hue="sex", palette="YlGnBu")

#sns.stripplot(x="day", y="tip", data=tips, hue="sex", palette="YlGnBu", dodge=True)

#sns.jointplot(x="tip", y="total_bill", data=tips, kind="hex", cmap="YlGnBu")

#sns.pairplot(titanic.select_dtypes(['number']), hue='pclass')

sns.heatmap(titanic.select_dtypes(include=['number']).corr(), annot=True, cmap="YlGnBu")

plt.show()

# Generate 100 random data points along 3 dimensions
x, y, scale = np.random.randn(3, 100)
fig, ax = plt.subplots()

# Map each onto a scatterplot we'll create with Matplotlib
ax.scatter(x=x, y=y, c=scale, s=np.abs(scale)*500)
ax.set(title="Some random data, created with JupyterLab!")
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Step 1: Generate sample signals (simulating two people talking)
np.random.seed(0)
time = np.linspace(0, 10, 1000)

s1 = np.sin(2 * time)         # Signal 1: Sine wave
s2 = np.sign(np.sin(3 * time))  # Signal 2: Square wave (like voice pulses)

S = np.c_[s1, s2]  # Stack signals column-wise
S = S + 0.2 * np.random.normal(size=S.shape)  # Add noise

# Step 2: Mix signals (simulating microphone recordings)
A = np.array([[1, 0.5], [0.5, 1]])  # Mixing matrix
X = np.dot(S, A.T)  # Mixed signals

# Step 3: Apply ICA to recover original signals
ica = FastICA(n_components=2)
S_ica = ica.fit_transform(X)  # Reconstructed signals

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original Signals (Sources)")
plt.plot(time, S)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend(["Signal 1", "Signal 2"])
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("Mixed Signals (Microphone Inputs)")
plt.plot(time, X)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend(["Mic 1", "Mic 2"])
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title("Recovered Signals using ICA")
plt.plot(time, S_ica)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend(["Recovered 1", "Recovered 2"])
plt.grid(True)

plt.tight_layout()
plt.show()

"""
# ...existing code...
"""
Fuzzy set utilities (example usage commented out above).
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

# ...existing code...
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
    p = concentration_g_per_100ml(weight_g, volume_ml)

    if p == 0:
        raise ValueError("Concentration is zero (no solute); specific rotation undefined")

    # Formula: [α] = 100 * α / (l * p)
    specific_alpha = (100.0 * alpha) / (l_dm * p)
    return specific_alpha

def concentration_g_per_100ml(weight_g: float, volume_ml: float) -> float:
    """Return concentration in g per 100 mL for given weight (g) and volume (mL)."""
    if volume_ml <= 0:
        raise ValueError("volume_ml must be > 0")
    if weight_g < 0:
        raise ValueError("weight_g must be >= 0")
    return (weight_g / volume_ml) * 100.0

__all__ = ["specific_rotation", "concentration_g_per_100ml", "fuzzy_union",
           "fuzzy_intersection", "fuzzy_complement", "fuzzy_difference",
           "fuzzy_relation", "maxmin_composition"]
# ...existing code...

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
    parser.add_argument("--precision", type=int, default=6, help="Decimal places for specific rotation output (default: 6)")
    args = parser.parse_args()

    try:
        result = specific_rotation(args.alpha, args.l_cm, args.weight_g, args.volume_ml)
    except ValueError as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)
    else:
        conc = concentration_g_per_100ml(args.weight_g, args.volume_ml)
        print("Specific Rotation Calculation")
        print("-----------------------------")
        print(f"Observed rotation (α): {args.alpha:.3f} °")
        print(f"Tube length (l):      {args.l_cm:.2f} cm")
        print(f"Mass of solute:       {args.weight_g:.3f} g")
        print(f"Volume of solution:   {args.volume_ml:.2f} mL")
        print(f"Concentration (p):    {conc:.6g} g/100mL")
        print()
        print(f"Specific Rotation [α]: {result:.{args.precision}f} ° · dm⁻¹ · (g/100mL)⁻¹")
# ...existing code...