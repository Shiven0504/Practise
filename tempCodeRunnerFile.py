# ...existing code...
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier


def fuzzy_temperature_demo(sample_temps=None, plot=True, save_plot=False, out_path="temperature_fuzzy_sets.png", verbose=False):
    """
    Demonstrate temperature fuzzy sets (cold, warm, hot), print a table of memberships
    for sample temperatures and optionally plot the membership functions.
    """
    def _trimf(x, abc):
        a, b, c = abc
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x, dtype=float)
        if b != a:
            idx = (x >= a) & (x <= b)
            y[idx] = (x[idx] - a) / (b - a)
        else:
            y[x == a] = 1.0
        if c != b:
            idx = (x >= b) & (x <= c)
            y[idx] = (c - x[idx]) / (c - b)
        else:
            y[x == c] = 1.0
        return np.clip(y, 0.0, 1.0)

    def _interp_membership(x, mf, value):
        return float(np.interp(value, x, mf))

    try:
        import skfuzzy as fuzz  # type: ignore
        trimf = fuzz.trimf
        interp_membership = fuzz.interp_membership
        if verbose:
            print("Using scikit-fuzzy for membership calculations.")
    except Exception:
        trimf = _trimf
        interp_membership = _interp_membership
        if verbose:
            print("scikit-fuzzy not available; using fallback implementations.")

    x_temp = np.arange(0, 41, 1)
    cold = trimf(x_temp, [0, 0, 20])
    warm = trimf(x_temp, [10, 20, 30])
    hot  = trimf(x_temp, [20, 40, 40])

    if sample_temps is None:
        sample_temps = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])

    cold_m = np.array([interp_membership(x_temp, cold, t) for t in sample_temps])
    warm_m = np.array([interp_membership(x_temp, warm, t) for t in sample_temps])
    hot_m  = np.array([interp_membership(x_temp, hot, t)  for t in sample_temps])

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
        plt.scatter(sample_temps, cold_m, c='C0', s=25, zorder=5)
        plt.scatter(sample_temps, warm_m, c='C1', s=25, zorder=5)
        plt.scatter(sample_temps, hot_m,  c='C2', s=25, zorder=5)
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


def run_nn_and_fuzzy(seed: int = 1, no_plot: bool = False, save_plot: str | None = None):
    np.random.seed(seed)

    # Neural network (AND function)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    try:
        nn = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, learning_rate_init=0.1,
                           solver='lbfgs', random_state=seed)
        nn.fit(X, y)
    except Exception as e:
        print("Error training MLPClassifier:", e)
        return

    preds = nn.predict(X)
    print("\n--- Neural Network Example ---")
    print("Predictions for AND gate:", preds)
    print("Expected:", y)
    print("Accuracy:", float((preds == y).mean()))

    # Fuzzy demo
    fuzzy_temperature_demo(plot=not no_plot, save_plot=bool(save_plot), out_path=save_plot or "temperature_fuzzy_sets.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NN AND demo and fuzzy temperature demo.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--no-plot", action="store_true", help="Do not show plots")
    parser.add_argument("--save-plot", type=str, default=None, help="Save fuzzy plot to given path")
    args = parser.parse_args()

    try:
        run_nn_and_fuzzy(seed=args.seed, no_plot=args.no_plot, save_plot=args.save_plot)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
# ...existing code...