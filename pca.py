"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# Load Dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 1️⃣ Principal Component Analysis (PCA)
# -------------------------
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
for i, target in enumerate(np.unique(y)):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], label=target_names[i])
plt.title("PCA - 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)

# -------------------------
# 2️⃣ Linear Discriminant Analysis (LDA)
# -------------------------
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

plt.figure(figsize=(7,5))
for i, target in enumerate(np.unique(y)):
    plt.scatter(X_lda[y == target, 0], X_lda[y == target, 1], label=target_names[i])
plt.title("LDA - 2D Projection")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()
"""

import pandas as pd
import numpy as np
from math import log2


def entropy(target_col):
    """Calculate Shannon entropy of a target column."""
    _, counts = np.unique(target_col, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def info_gain(data, split_attribute, target_name="Play"):
    """Calculate information gain for a given attribute split."""
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_attribute], return_counts=True)
    total = counts.sum()

    weighted_entropy = sum(
        (counts[i] / total) * entropy(data[data[split_attribute] == values[i]][target_name])
        for i in range(len(values))
    )

    return total_entropy - weighted_entropy


def id3(data, original_data, features, target_name="Play", parent_node_class=None):
    """Build a decision tree using the ID3 algorithm (recursive)."""
    unique_targets = np.unique(data[target_name])

    # Pure node — all samples have the same class
    if len(unique_targets) == 1:
        return unique_targets[0]

    # No samples left — return majority class from original data
    if len(data) == 0:
        vals, counts = np.unique(original_data[target_name], return_counts=True)
        return vals[np.argmax(counts)]

    # No features left — return current majority class
    if len(features) == 0:
        return parent_node_class

    # Determine majority class of current subset
    vals, counts = np.unique(data[target_name], return_counts=True)
    parent_node_class = vals[np.argmax(counts)]

    # Select feature with highest information gain
    gains = [info_gain(data, f, target_name) for f in features]
    best_feature = features[np.argmax(gains)]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in np.unique(data[best_feature]):
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = id3(
            subset, original_data, remaining_features, target_name, parent_node_class
        )

    return tree


# Sample dataset (classic weather/play-tennis problem)
data = {
    "Outlook":     ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
                    "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
                    "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity":    ["High", "High", "High", "High", "Normal", "Normal", "Normal",
                    "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind":        ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
                    "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "Play":        ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                    "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"],
}

df = pd.DataFrame(data)
features = ["Outlook", "Temperature", "Humidity", "Wind"]

tree = id3(df, df, features)
print("Decision Tree (ID3):")
print(tree)

