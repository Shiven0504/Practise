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
import matplotlib.pyplot as plt
from typing import Tuple

def analyze_employee_data(df: pd.DataFrame, plot: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Print summary and group-wise aggregations for employee data.

    Expects columns: Name, Department, Salary, Experience.
    Returns (avg_salary_by_dept, agg_results_df).
    """
    required = {"Name", "Department", "Salary", "Experience"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    if df.empty:
        print("DataFrame is empty.")
        return pd.Series(dtype=float), pd.DataFrame()

    print("=== Original DataFrame ===")
    print(df.to_string(index=False), "\n")

    # 1. Overall Aggregation
    print("=== Overall Salary Statistics ===")
    overall = df["Salary"].agg(["mean", "max", "min"]).rename({"mean": "mean", "max": "max", "min": "min"})
    print(overall.to_frame().T.round(2), "\n")

    # 2. Group-wise Aggregation (Sorted by Average Salary)
    print("=== Average Salary by Department (Sorted) ===")
    avg_salary = df.groupby("Department")["Salary"].mean().sort_values(ascending=False)
    print(avg_salary.round(2), "\n")

    # 3. Multiple Aggregations per Group
    print("=== Aggregated Salary & Experience by Department ===")
    agg_results = (
        df.groupby("Department")
          .agg(
              Avg_Salary=("Salary", "mean"),
              Max_Salary=("Salary", "max"),
              Min_Salary=("Salary", "min"),
              Avg_Experience=("Experience", "mean")
          )
          .sort_values(by="Avg_Salary", ascending=False)
          .round(2)
    )
    print(agg_results, "\n")

    # 4. Employee Count per Department
    print("=== Employee Count by Department ===")
    counts = df["Department"].value_counts()
    print(counts, "\n")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        avg_salary.plot(kind="bar", ax=ax[0], title="Avg Salary by Department")
        agg_results[["Avg_Experience"]].plot(kind="bar", ax=ax[1], title="Avg Experience by Department", legend=False)
        plt.tight_layout()
        plt.show()

    return avg_salary, agg_results

# ---- Sample Dataset ----
if __name__ == "__main__":
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
        "Department": ["HR", "IT", "IT", "Finance", "HR", "Finance"],
        "Salary": [50000, 60000, 55000, 65000, 52000, 70000],
        "Experience": [2, 5, 3, 7, 4, 10]
    }

    df = pd.DataFrame(data)
    avg_salary, agg_results = analyze_employee_data(df, plot=False)