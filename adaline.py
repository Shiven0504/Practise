
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
