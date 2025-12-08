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


# ...existing code...
import pandas as pd
import numpy as np
from math import log2
from typing import Dict, Any, List, Optional

# Step 1: Entropy Function
def entropy(target_col: pd.Series) -> float:
    values, counts = np.unique(target_col, return_counts=True)
    probs = counts / counts.sum()
    # avoid log2(0) by filtering zero probabilities
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))

# Step 2: Information Gain
def info_gain(data: pd.DataFrame, split_attribute: str, target_name: str = "Play") -> float:
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_attribute], return_counts=True)
    total_count = len(data)
    weighted_entropy = 0.0
    for v, c in zip(values, counts):
        subset = data[data[split_attribute] == v]
        weighted_entropy += (c / total_count) * entropy(subset[target_name])
    return total_entropy - weighted_entropy

# Step 3: ID3 Algorithm
def id3(data: pd.DataFrame,
        original_data: pd.DataFrame,
        features: List[str],
        target_name: str = "Play",
        parent_node_class: Optional[Any] = None) -> Any:
    # If all target values are the same, return that value
    unique_targets = np.unique(data[target_name])
    if len(unique_targets) == 1:
        return unique_targets[0]

    # If dataset is empty, return the most common target in the original dataset
    if data.shape[0] == 0:
        return original_data[target_name].value_counts().idxmax()

    # If no features left, return the parent node class (majority class)
    if len(features) == 0:
        return parent_node_class

    # Set parent node class to the majority class of current node
    parent_node_class = data[target_name].value_counts().idxmax()

    # Compute information gain for each feature and select the best
    gains = [info_gain(data, feature, target_name) for feature in features]
    best_feature = features[int(np.argmax(gains))]

    tree: Dict[str, Any] = {best_feature: {}}

    # For each possible value of the best feature, grow subtree
    for value in np.unique(data[best_feature]):
        sub_data = data[data[best_feature] == value]
        remaining_features = [f for f in features if f != best_feature]
        subtree = id3(sub_data, original_data, remaining_features, target_name, parent_node_class)
        tree[best_feature][value] = subtree

    return tree

# Utility: pretty-print the tree
def print_tree(tree: Any, indent: str = "") -> None:
    if not isinstance(tree, dict):
        print(indent + str(tree))
        return
    for feature, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}{feature} = {value} ->", end=" ")
            if isinstance(subtree, dict):
                print()
                print_tree(subtree, indent + "    ")
            else:
                print(subtree)

# Step 4: Sample Dataset (Weather Data)
def main():
    data = {
        'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
        'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
        'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
        'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
        'Play': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
    }

    df = pd.DataFrame(data)

    # Step 5: Train ID3 Tree
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    tree = id3(df, df, features)

    # Output the tree
    print("Generated Decision Tree (ID3):")
    print_tree(tree)

if __name__ == "__main__":
    main()