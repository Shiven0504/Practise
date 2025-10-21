
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


from sklearn.neural_network import MLPClassifier
import numpy as np

# Example: AND function with Neural Network
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # AND output

# Simple neural network with 1 hidden layer of 2 neurons
nn = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, learning_rate_init=0.1)
nn.fit(X, y)

print("\n--- Neural Network Example ---")
print("Predictions for AND gate:", nn.predict(X))


# 2. Fuzzy Logic (using scikit-fuzzy)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

# --- 1) Neural network (AND) ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND output

# Deterministic MLP for reproducibility
nn = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, learning_rate_init=0.1, random_state=42, tol=1e-6)
nn.fit(X, y)

pred = nn.predict(X)
acc = accuracy_score(y, pred)
print("\n--- Neural Network Example ---")
print("Predictions for AND gate:", pred)
print(f"Training accuracy: {acc:.2f}")

# show probabilities if available
if hasattr(nn, "predict_proba"):
    probs = nn.predict_proba(X)
    print("Predicted probabilities:\n", np.round(probs, 3))
else:
    print("predict_proba not available for this estimator.")

# --- 2) Fuzzy Logic (using scikit-fuzzy) ---
# Define universe and membership functions for 'temperature' (0..40 °C)
x_temp = np.arange(0, 41, 1)
cold = fuzz.trimf(x_temp, [0, 0, 20])
warm = fuzz.trimf(x_temp, [10, 20, 30])
hot  = fuzz.trimf(x_temp, [20, 40, 40])

# Evaluate memberships for several sample temperatures
sample_temps = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
cold_m = np.array([fuzz.interp_membership(x_temp, cold, t) for t in sample_temps])
warm_m = np.array([fuzz.interp_membership(x_temp, warm, t) for t in sample_temps])
hot_m  = np.array([fuzz.interp_membership(x_temp, hot, t)  for t in sample_temps])

print("\n--- Fuzzy Logic Example ---")
print("Temp | cold  | warm  | hot")
for t, c, w, h in zip(sample_temps, cold_m, warm_m, hot_m):
    print(f"{t:4d} | {c:5.3f} | {w:5.3f} | {h:5.3f}")

# Optional: plot membership functions
plt.figure(figsize=(6, 3.5))
plt.plot(x_temp, cold, label="cold")
plt.plot(x_temp, warm, label="warm")
plt.plot(x_temp, hot, label="hot")
plt.scatter(sample_temps, cold_m, c='C0', s=20)
plt.scatter(sample_temps, warm_m, c='C1', s=20)
plt.scatter(sample_temps, hot_m,  c='C2', s=20)
plt.xlabel("Temperature (°C)")
plt.ylabel("Membership degree")
plt.title("Temperature fuzzy sets")
plt.legend()
plt.tight_layout()
plt.show()