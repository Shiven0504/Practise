
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
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Dict, Any

from sklearn.neural_network import MLPClassifier


def fuzzy_temperature_demo(
    sample_temps: Optional[Sequence[int]] = None,
    plot: bool = True,
    save_plot: bool = False,
    out_path: str = "temperature_fuzzy_sets.png",
) -> Dict[str, Any]:
    """
    Demonstrate temperature fuzzy sets (cold, warm, hot), print a table of memberships
    for sample temperatures and optionally plot the membership functions.
    Returns the membership arrays and x axis for further use.
    """
    def _trimf(x: np.ndarray, abc: Sequence[float]) -> np.ndarray:
        a, b, c = abc
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x, dtype=float)
        # rising edge a..b
        if b != a:
            idx = (x >= a) & (x <= b)
            y[idx] = (x[idx] - a) / (b - a)
        else:
            y[x == a] = 1.0
        # falling edge b..c
        if c != b:
            idx = (x >= b) & (x <= c)
            y[idx] = (c - x[idx]) / (c - b)
        else:
            y[x == c] = 1.0
        return np.clip(y, 0.0, 1.0)

    def _interp_membership(x: np.ndarray, mf: np.ndarray, value: float) -> float:
        return float(np.interp(value, x, mf))

    # prefer scikit-fuzzy if available
    try:
        import skfuzzy as fuzz  # type: ignore
        trimf = fuzz.trimf
        interp_membership = fuzz.interp_membership
    except Exception:
        trimf = _trimf
        interp_membership = _interp_membership

    x_temp = np.arange(0, 41, 1)  # universe: 0..40 °C
    cold = trimf(x_temp, [0, 0, 20])
    warm = trimf(x_temp, [10, 20, 30])
    hot = trimf(x_temp, [20, 40, 40])

    if sample_temps is None:
        sample_temps = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])

    cold_m = np.array([interp_membership(x_temp, cold, float(t)) for t in sample_temps])
    warm_m = np.array([interp_membership(x_temp, warm, float(t)) for t in sample_temps])
    hot_m = np.array([interp_membership(x_temp, hot, float(t)) for t in sample_temps])

    # Print a tidy table
    print("\n--- Fuzzy Logic Example (Temperature) ---")
    print("Temp |  cold  |  warm  |   hot ")
    print("--------------------------------")
    for t, c, w, h in zip(sample_temps, cold_m, warm_m, hot_m):
        print(f"{int(t):4d} | {c:6.3f} | {w:6.3f} | {h:6.3f}")

    if plot:
        plt.figure(figsize=(7, 3.5))
        plt.plot(x_temp, cold, label="cold", lw=2)
        plt.plot(x_temp, warm, label="warm", lw=2)
        plt.plot(x_temp, hot, label="hot", lw=2)
        plt.scatter(sample_temps, cold_m, c="C0", s=25, zorder=5)
        plt.scatter(sample_temps, warm_m, c="C1", s=25, zorder=5)
        plt.scatter(sample_temps, hot_m, c="C2", s=25, zorder=5)
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

    return {
        "x": x_temp,
        "cold": cold,
        "warm": warm,
        "hot": hot,
        "sample_temps": np.asarray(sample_temps),
        "cold_m": cold_m,
        "warm_m": warm_m,
        "hot_m": hot_m,
    }


def run_nn_and_fuzzy(plot: bool = True, save_plot: bool = False, out_path: str = "temperature_fuzzy_sets.png") -> Dict[str, Any]:
    """
    Train a tiny NN to learn the AND function and run the fuzzy temperature demo.
    Returns a dict with predictions and fuzzy results.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND output

    # lbfgs ignores learning_rate_init; removed to avoid warnings
    nn = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, solver="lbfgs", random_state=1)
    nn.fit(X, y)

    preds = nn.predict(X)
    acc = float((preds == y).mean())

    print("\n--- Neural Network Example ---")
    print("Predictions for AND gate:", preds)
    print("Expected:", y)
    print("Accuracy:", acc)

    fuzzy_res = fuzzy_temperature_demo(plot=plot, save_plot=save_plot, out_path=out_path)

    return {"preds": preds, "accuracy": acc, "fuzzy": fuzzy_res}


def main():
    p = argparse.ArgumentParser(description="Run NN AND example and fuzzy temperature demo.")
    p.add_argument("--no-plot", action="store_true", help="Do not show plots (useful for headless environments).")
    p.add_argument("--