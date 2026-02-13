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
from typing import Any, Dict, List, Optional, Tuple, Union

# Step 1: Entropy Function (robust to zero counts)
def entropy(target_col: Union[np.ndarray, List[Any]]) -> float:
    values, counts = np.unique(target_col, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

# Step 2: Information Gain
def info_gain(data: pd.DataFrame, split_attribute: str, target_name: str = "Play") -> float:
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_attribute], return_counts=True)
    total = counts.sum()
    weighted_entropy = 0.0
    for v, c in zip(values, counts):
        subset = data[data[split_attribute] == v][target_name]
        weighted_entropy += (c / total) * entropy(subset)
    return float(total_entropy - weighted_entropy)

# Step 3: ID3 Algorithm (returns nested dict)
def id3(
    data: pd.DataFrame,
    original_data: pd.DataFrame,
    features: List[str],
    target_name: str = "Play",
    parent_node_class: Optional[Any] = None
) -> Any:
    # If all targets have the same value, return it (leaf)
    if len(np.unique(data[target_name])) == 1:
        return np.unique(data[target_name])[0]

    # If dataset becomes empty, return most common target in original data
    if data.shape[0] == 0:
        return original_data[target_name].mode()[0]

    # If no features left, return parent node's class
    if len(features) == 0:
        return parent_node_class

    # Set parent node class to the most common class of current node
    parent_node_class = data[target_name].mode()[0]

    # Compute information gain for each feature and choose the best
    gains = [info_gain(data, feature, target_name) for feature in features]
    best_feature = features[int(np.argmax(gains))]

    tree: Dict[str, Dict[Any, Any]] = {best_feature: {}}
    for value in np.unique(data[best_feature]):
        sub_data = data[data[best_feature] == value].reset_index(drop=True)
        remaining_features = [f for f in features if f != best_feature]
        subtree = id3(sub_data, original_data, remaining_features, target_name, parent_node_class)
        tree[best_feature][value] = subtree

    return tree

# Utility: classify a single sample using the learned tree
def classify(tree: Dict[str, Any], sample: Dict[str, Any]) -> Any:
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    if root not in sample:
        # missing attribute: cannot traverse, return None
        return None
    value = sample[root]
    branch = tree[root].get(value)
    if branch is None:
        # unseen attribute value: return None
        return None
    return classify(branch, sample)

# Utility: pretty-print tree
def print_tree(tree: Any, depth: int = 0) -> None:
    if not isinstance(tree, dict):
        print("  " * depth + f"-> {tree}")
        return
    for feature, branches in tree.items():
        for val, subtree in branches.items():
            print("  " * depth + f"{feature} = {val}:")
            print_tree(subtree, depth + 1)

# Step 4: Sample Dataset (Weather Data)
def build_weather_df() -> pd.DataFrame:
    data = {
        'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
        'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
        'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
        'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
        'Play': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
    }
    return pd.DataFrame(data)

# Step 5: Run ID3 demo
def run_id3_demo() -> None:
    df = build_weather_df()
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    tree = id3(df, df, features, target_name="Play")
    print("Generated Decision Tree (ID3):")
    print_tree(tree)
    # Example classification of the training set (demonstration)
    print("\nPredictions on training samples:")
    for idx, row in df.iterrows():
        pred = classify(tree, row.to_dict())
        print(f"Sample {idx}: True={row['Play']} Pred={pred}")

# Optional: PCA / LDA demo (if sklearn is available)
def run_pca_lda_demo() -> None:
    try:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        import matplotlib.pyplot as plt
    except Exception as e:
        print("sklearn or matplotlib not available — skipping PCA/LDA demo.")
        return

    data = load_iris()
    X = data.data
    y = data.target
    target_names = data.target_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6,4))
    for i, target in enumerate(np.unique(y)):
        plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], label=target_names[i])
    plt.title("PCA - 2D Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()
    print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)

    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)

    plt.figure(figsize=(6,4))
    for i, target in enumerate(np.unique(y)):
        plt.scatter(X_lda[y == target, 0], X_lda[y == target, 1], label=target_names[i])
    plt.title("LDA - 2D Projection")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_id3_demo()
    # call PCA/LDA demo optionally
    run_pca_lda_demo()
# ...existing code...