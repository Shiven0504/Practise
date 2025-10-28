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

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from deap import base, creator, tools, algorithms 
 
def total_distance(tour, dist_matrix): 
    distance = 0 
    for i in range(len(tour) - 1): 
        distance += dist_matrix[tour[i]][tour[i + 1]] 
    distance += dist_matrix[tour[-1]][tour[0]]  # Return to starting city 
    return distance 
 
num_cities = 10  # Number of cities 
np.random.seed(1) 
cities = np.random.rand(num_cities, 2) * 100 
dist_matrix = np.linalg.norm(cities[:, np.newaxis] - cities[np.newaxis, :], axis=2) 
 
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) 
creator.create("Individual", list, fitness=creator.FitnessMin) 
 
toolbox = base.Toolbox() 
toolbox.register("indices", random.sample, range(num_cities), num_cities) 
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices) 
toolbox.register("population", tools.initRepeat, list, toolbox.individual) 
 
def evaluate(individual): 
    return (total_distance(individual, dist_matrix),) 
 
toolbox.register("evaluate", evaluate) 
toolbox.register("mate", tools.cxOrdered) 
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) 
toolbox.register("select", tools.selTournament, tournsize=3) 
 
print("Starting Genetic Algorithm for TSP...\n") 
 
population = toolbox.population(n=100) 
NGEN = 20    
CXPB, MUTPB = 0.7, 0.2 
 
stats = tools.Statistics(lambda ind: ind.fitness.values) 
stats.register("avg", np.mean) 
stats.register("min", np.min) 
stats.register("max", np.max) 
 
pop, log = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, 
                               ngen=NGEN, stats=stats, verbose=True) 
 
best_ind = tools.selBest(population, 1)[0] 
min_dist = total_distance(best_ind, dist_matrix) 
 
print("\n--- GA Finished ---") 
print("Best tour found:") 
print(best_ind) 
print(f"Minimum tour distance: {min_dist:.2f}") 
 
plt.figure(figsize=(7, 6)) 
plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=50, label="Cities") 
tour_coords = cities[best_ind + [best_ind[0]]]  # Return to start 
plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'r-', linewidth=1.5, label="Best Path") 
plt.title(f"Best Tour Found (Distance: {min_dist:.2f})") 
plt.xlabel("X Coordinate") 
plt.ylabel("Y Coordinate") 
plt.legend() 
plt.grid(True) 
plt.show()