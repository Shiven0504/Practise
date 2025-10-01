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

# Step 1: Entropy Function
def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    entropy_val = -sum((counts[i]/sum(counts)) * log2(counts[i]/sum(counts)) for i in range(len(values)))
    return entropy_val

# Step 2: Information Gain
def info_gain(data, split_attribute, target_name="Play"):
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_attribute], return_counts=True)
    
    weighted_entropy = sum((counts[i]/sum(counts)) * entropy(data[data[split_attribute] == values[i]][target_name]) 
                           for i in range(len(values)))
    
    return total_entropy - weighted_entropy

# Step 3: ID3 Algorithm
def id3(data, original_data, features, target_name="Play", parent_node_class=None):
    if len(np.unique(data[target_name])) == 1:  # Pure node
        return np.unique(data[target_name])[0]
    
    if len(data) == 0:  # No samples
        return np.unique(original_data[target_name])[np.argmax(
            np.unique(original_data[target_name], return_counts=True)[1])]
    
    if len(features) == 0:  # No features left
        return parent_node_class
    
    parent_node_class = np.unique(data[target_name])[np.argmax(
        np.unique(data[target_name], return_counts=True)[1])]
    
    gains = [info_gain(data, feature, target_name) for feature in features]
    best_feature = features[np.argmax(gains)]
    
    tree = {best_feature: {}}
    
    for value in np.unique(data[best_feature]):
        sub_data = data[data[best_feature] == value]
        new_features = [f for f in features if f != best_feature]
        
        subtree = id3(sub_data, original_data, new_features, target_name, parent_node_class)
        tree[best_feature][value] = subtree
    
    return tree

# Step 4: Sample Dataset (Weather Data)
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
print(tree)
