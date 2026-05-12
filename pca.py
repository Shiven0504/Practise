"""
Archived: PCA + LDA on Iris dataset using sklearn.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

data = load_iris()
X, y, names = data.data, data.target, data.target_names
X = StandardScaler().fit_transform(X)

for model, title, labels in [
    (PCA(n_components=2),  "PCA", ("PC1", "PC2")),
    (LDA(n_components=2),  "LDA", ("LD1", "LD2")),
]:
    X2 = model.fit_transform(X) if title == "PCA" else model.fit_transform(X, y)
    plt.figure(figsize=(7, 5))
    for i in np.unique(y):
        plt.scatter(X2[y==i, 0], X2[y==i, 1], label=names[i])
    plt.title(f"{title} - 2D Projection")
    plt.xlabel(labels[0]); plt.ylabel(labels[1])
    plt.legend(); plt.show()
    if title == "PCA":
        print("Explained Variance Ratio (PCA):", model.explained_variance_ratio_)
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from numpy.typing import NDArray


def majority(col: Union[pd.Series, NDArray]) -> Any:
    """
    Find the most frequent value in a column.
    
    Args:
        col: Series or array of values
    
    Returns:
        The most frequently occurring value
    """
    vals, counts = np.unique(col, return_counts=True)
    return vals[np.argmax(counts)]


def entropy(col: Union[pd.Series, NDArray]) -> float:
    """
    Calculate Shannon entropy of a column.
    
    Args:
        col: Series or array of class labels
    
    Returns:
        Entropy value (float)
    """
    _, counts = np.unique(col, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))


def info_gain(data: pd.DataFrame, feature: str, target: str = "Play") -> float:
    """
    Calculate information gain for a feature in ID3 decision tree.
    
    Args:
        data: DataFrame containing the dataset
        feature: Column name of the feature to evaluate
        target: Column name of the target variable (default: "Play")
    
    Returns:
        Information gain value (float) - higher means more discriminative
    """
    total = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted = sum(
        (counts[i] / counts.sum()) * entropy(data[data[feature] == v][target])
        for i, v in enumerate(values)
    )
    return total - weighted


def id3(
    data: pd.DataFrame,
    original_data: pd.DataFrame,
    features: List[str],
    target: str = "Play",
    parent_class: Optional[Any] = None
) -> Union[Any, Dict]:
    """
    Build a decision tree using the ID3 (Iterative Dichotomiser 3) algorithm.
    
    Args:
        data: Current dataset (subset of original during recursion)
        original_data: Full original dataset for majority voting fallback
        features: List of available feature column names
        target: Column name of the target variable (default: "Play")
        parent_class: Most common class value from parent node (for leaf fallback)
    
    Returns:
        If leaf: the class label (Any)
        If branch: dictionary mapping feature values to subtrees (Dict)
    """
    unique = np.unique(data[target])

    if len(unique) == 1:
        return unique[0]
    if len(data) == 0:
        return majority(original_data[target])
    if len(features) == 0:
        return parent_class

    current_majority = majority(data[target])
    best = features[np.argmax([info_gain(data, f, target) for f in features])]
    remaining = [f for f in features if f != best]

    return {best: {
        v: id3(data[data[best] == v], original_data, remaining, target, current_majority)
        for v in np.unique(data[best])
    }}


data = {
    "Outlook":     ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast",
                    "Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool",
                    "Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity":    ["High","High","High","High","Normal","Normal","Normal",
                    "High","Normal","Normal","Normal","High","Normal","High"],
    "Wind":        ["Weak","Strong","Weak","Weak","Weak","Strong","Strong",
                    "Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "Play":        ["No","No","Yes","Yes","Yes","No","Yes",
                    "No","Yes","Yes","Yes","Yes","Yes","No"],
}

df = pd.DataFrame(data)
features = ["Outlook", "Temperature", "Humidity", "Wind"]

tree = id3(df, df, features)
print("Decision Tree (ID3):")
print(tree)
