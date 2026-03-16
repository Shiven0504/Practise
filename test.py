"""
Archived: seaborn plots (heatmap, scatter) and ICA signal separation.
See git history if you need these.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

try:
    import skfuzzy as fuzz
    trimf = fuzz.trimf
    interp_membership = fuzz.interp_membership
except ImportError:
    def trimf(x, abc):
        a, b, c = abc
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        if b != a:
            m = (x >= a) & (x <= b)
            y[m] = (x[m] - a) / (b - a)
        if c != b:
            m = (x >= b) & (x <= c)
            y[m] = (c - x[m]) / (c - b)
        return np.clip(y, 0.0, 1.0)

    def interp_membership(x, mf, value):
        return float(np.interp(value, x, mf))


def fuzzy_temperature_demo():
    x = np.arange(0, 41)
    cold = trimf(x, [0, 0, 20])
    warm = trimf(x, [10, 20, 30])
    hot  = trimf(x, [20, 40, 40])

    temps = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    cold_m = [interp_membership(x, cold, t) for t in temps]
    warm_m = [interp_membership(x, warm, t) for t in temps]
    hot_m  = [interp_membership(x, hot,  t) for t in temps]

    print("\n--- Fuzzy Temperature ---")
    print("Temp | cold  | warm  | hot")
    print("-----------------------------")
    for t, c, w, h in zip(temps, cold_m, warm_m, hot_m):
        print(f"{t:4d} | {c:.3f} | {w:.3f} | {h:.3f}")

    plt.figure(figsize=(7, 3.5))
    for mf, label in zip([cold, warm, hot], ["cold", "warm", "hot"]):
        plt.plot(x, mf, label=label, lw=2)
    plt.scatter(temps, cold_m, c="C0", s=25, zorder=5)
    plt.scatter(temps, warm_m, c="C1", s=25, zorder=5)
    plt.scatter(temps, hot_m,  c="C2", s=25, zorder=5)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Membership degree")
    plt.title("Temperature fuzzy sets")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def mlp_and_gate():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0, 0, 0, 1])

    nn = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000,
                       learning_rate_init=0.1, solver="lbfgs", random_state=1)
    nn.fit(X, y)

    preds = nn.predict(X)
    print("\n--- MLP AND Gate ---")
    print("Predictions:", preds)
    print("Expected:   ", y)
    print("Accuracy:   ", (preds == y).mean())


if __name__ == "__main__":
    mlp_and_gate()
    fuzzy_temperature_demo()
