"""
import numpy as np 
import skfuzzy as fuzz 
from skfuzzy import control as ctrl 
import matplotlib.pyplot as plt 
 
TempError = ctrl.Antecedent(np.arange(-5, 6, 1), 'TempError') 
Humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'Humidity') 
CompressorSpeed = ctrl.Consequent(np.arange(0, 101, 1), 'CompressorSpeed') 
 
TempError['Cold'] = fuzz.trimf(TempError.universe, [-5, -5, 0]) 
TempError['Good'] = fuzz.trimf(TempError.universe, [-1, 0, 1]) 
TempError['Hot'] = fuzz.trimf(TempError.universe, [0, 5, 5]) 
 
Humidity['Low'] = fuzz.trimf(Humidity.universe, [0, 0, 50]) 
Humidity['High'] = fuzz.trimf(Humidity.universe, [50, 100, 100]) 
 
CompressorSpeed['Off'] = fuzz.trimf(CompressorSpeed.universe, [0, 0, 10]) 
CompressorSpeed['Slow'] = fuzz.trimf(CompressorSpeed.universe, [10, 30, 50]) 
CompressorSpeed['Medium'] = fuzz.trimf(CompressorSpeed.universe, [40, 60, 80]) 
CompressorSpeed['Fast'] = fuzz.trimf(CompressorSpeed.universe, [70, 100, 100]) 
 
rule1 = ctrl.Rule(TempError['Cold'], CompressorSpeed['Off']) 
rule2 = ctrl.Rule(TempError['Good'], CompressorSpeed['Slow']) 
rule3 = ctrl.Rule(TempError['Hot'] & Humidity['Low'], CompressorSpeed['Medium']) 
rule4 = ctrl.Rule(TempError['Hot'] & Humidity['High'], CompressorSpeed['Fast']) 
 
ac_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4]) 
ac_sim = ctrl.ControlSystemSimulation(ac_ctrl) 
# ...existing code...
if __name__ == "__main__":
    print("\n--- Evaluating Controller ---") 
 
    ac_sim.input['TempError'] = 3 
    ac_sim.input['Humidity'] = 30 
    ac_sim.compute() 
    print(f"Case 1 (Hot, Low Humidity): Compressor Speed = {ac_sim.output['CompressorSpeed']:.2f}%") 
 
    ac_sim.input['TempError'] = 3 
    ac_sim.input['Humidity'] = 80 
    ac_sim.compute() 
    print(f"Case 2 (Hot, High Humidity): Compressor Speed = {ac_sim.output['CompressorSpeed']:.2f}%") 
 
    ac_sim.input['TempError'] = -2 
    ac_sim.input['Humidity'] = 50 
    ac_sim.compute() 
    print(f"Case 3 (Too Cold): Compressor Speed = {ac_sim.output['CompressorSpeed']:.2f}%") 
 
    fig, axs = plt.subplots(2, 2, figsize=(10, 7)) 
 
    TempError.view(ax=axs[0, 0]) 
    axs[0, 0].set_title('Temperature Error Membership Functions') 
 
    Humidity.view(ax=axs[0, 1]) 
    axs[0, 1].set_title('Humidity Membership Functions') 
 
    # show compressor membership functions and mark last computed output
    CompressorSpeed.view(ax=axs[1, 0]) 
    axs[1, 0].set_title('Compressor Speed Membership Functions') 
    # mark the last computed crisp output on the membership plot (Case 3)
    try:
        axs[1, 0].axvline(ac_sim.output['CompressorSpeed'], color='r', linestyle='--', label='Output')
        axs[1, 0].legend()
    except Exception:
        pass
 
    # hide unused subplot
    axs[1, 1].axis('off')
 
    plt.tight_layout() 
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