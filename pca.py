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
