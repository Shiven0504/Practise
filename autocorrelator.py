import numpy as np 
 
A1 = np.array([-1,  1, -1,  1]) 
A2 = np.array([ 1,  1,  1, -1]) 
A3 = np.array([-1, -1, -1,  1]) 
stored_patterns = [A1, A2, A3] 
 
W = np.zeros((4, 4)) 
for A in stored_patterns: 
    W += np.outer(A, A) 
 
np.fill_diagonal(W, 0) 
 
print("Weight Matrix (W):") 
print(W) 
 
def activation(x): 
    """Bipolar step activation function.""" 
    return np.where(x >= 0, 1, -1) 
Ax = np.array([-1,  1, -1,  1]) 
Ay = np.array([ 1,  1,  1,  1]) 
Az = np.array([-1, -1, -1, -1]) 
test_patterns = [Ax, Ay, Az] 
pattern_names = ["Ax", "Ay", "Az"] 
 
print("\n--- Testing Network Recall ---") 
for name, pattern in zip(pattern_names, test_patterns): 
    print(f"\nTesting with pattern: {name}") 
    print("Input:") 
    print(pattern) 
 
    output = activation(np.dot(pattern, W)) 
 
    print("Output after one step:") 
    print(output) 
 
    if np.array_equal(output, A1): 
        print("Result: Converged to pattern A1") 
    elif np.array_equal(output, A2): 
        print("Result: Converged to pattern A2") 
    elif np.array_equal(output, A3): 
        print("Result: Converged to pattern A3") 
    else: 
        print("Result: Did not converge to a stored pattern")