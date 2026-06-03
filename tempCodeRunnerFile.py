"""
Hopfield Associative Memory Network

A recurrent neural network for pattern storage and recall using the outer-product
learning rule (Hebb rule). Implements synchronous iterative recall with energy tracking.
"""
from typing import Tuple, Optional, List
import numpy as np

# Define training patterns (stored memories)
A1 = np.array([-1,  1, -1,  1], dtype=np.int8)
A2 = np.array([ 1,  1,  1, -1], dtype=np.int8)
A3 = np.array([-1, -1, -1,  1], dtype=np.int8)
stored_patterns = [A1, A2, A3]
pattern_names = ["A1", "A2", "A3"]

# Compute weight matrix using Hebb rule (outer product)
W = np.zeros((4, 4), dtype=np.float32)
for A in stored_patterns:
    W += np.outer(A, A)

np.fill_diagonal(W, 0)

print("Weight Matrix (W):")
print(W)
print()


def activation(x: np.ndarray) -> np.ndarray:
    """
    Bipolar step activation function.
    
    Args:
        x: Input array
        
    Returns:
        Array with values mapped to {-1, 1}
    """
    return np.where(x >= 0, 1, -1)


def energy(state: np.ndarray, W: np.ndarray) -> float:
    """
    Calculate Hopfield network energy (Lyapunov function).
    
    Energy decreases with each update step. At convergence, energy is minimized.
    
    Args:
        state: Current network state
        W: Weight matrix
        
    Returns:
        Network energy value
    """
    return -0.5 * state @ W @ state


def recall(
    pattern: np.ndarray,
    W: np.ndarray,
    activation_fn: callable,
    max_steps: int = 10,
    track_energy: bool = False
) -> Tuple[np.ndarray, int, bool, Optional[List[float]]]:
    """
    Synchronous iterative recall with optional energy tracking.
    
    Applies the update rule: state_t+1 = activation(state_t @ W)
    Updates continue until convergence or max_steps reached.
    
    Args:
        pattern: Initial pattern (possibly noisy)
        W: Weight matrix
        activation_fn: Activation function to apply
        max_steps: Maximum iteration steps
        track_energy: If True, return list of energy values at each step
        
    Returns:
        Tuple of (final_state, steps_taken, converged, energy_history)
    """
    state = pattern.copy().astype(np.float32)
    energy_history = [] if track_energy else None
    
    for step in range(1, max_steps + 1):
        if track_energy:
            energy_history.append(energy(state, W))
            
        new_state = activation_fn(np.dot(state, W))
        if np.array_equal(new_state, state):
            return new_state, step, True, energy_history
        state = new_state
    
    return state, max_steps, False, energy_history


def match_stored(
    state: np.ndarray,
    stored: List[np.ndarray],
    names: Optional[List[str]] = None
) -> Tuple[Optional[str], int]:
    """
    Match output state to a stored pattern.
    
    Args:
        state: Network output state
        stored: List of stored patterns
        names: Optional list of pattern names
        
    Returns:
        Tuple of (pattern_name_or_none, pattern_index_or_minus1)
    """
    for i, p in enumerate(stored):
        if np.array_equal(state, p):
            return (names[i] if names else i, i)
    return (None, -1)


def flip_bits(
    pattern: np.ndarray,
    num_flips: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create a noisy version of the pattern by flipping bits.
    
    Args:
        pattern: Original bipolar pattern
        num_flips: Number of bits to flip
        seed: Random seed for reproducibility
        
    Returns:
        Noisy version of the pattern
    """
    rng = np.random.default_rng(seed)
    noisy = pattern.copy()
    if num_flips > 0:
        indices = rng.choice(len(pattern), size=num_flips, replace=False)
        noisy[indices] *= -1
    return noisy


# Test patterns
Ax = np.array([-1,  1, -1,  1], dtype=np.int8)
Ay = np.array([ 1,  1,  1,  1], dtype=np.int8)
Az = np.array([-1, -1, -1, -1], dtype=np.int8)
test_patterns = [Ax, Ay, Az]
test_names = ["Ax", "Ay", "Az"]


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HOPFIELD NETWORK PATTERN RECALL DEMONSTRATION")
    print("="*60)
    
    print("\n--- Testing Network Recall (iterative) ---")
    for name, pattern in zip(test_names, test_patterns):
        print(f"\nTesting with pattern: {name}")
        print("Input:")
        print(pattern)

        final, steps, converged, energy_hist = recall(
            pattern, W, activation, max_steps=20, track_energy=True
        )

        print(f"Output after {steps} step(s):")
        print(final)
        
        if energy_hist:
            print(f"Energy progression: {[f'{e:.2f}' for e in energy_hist[:5]]}")
        
        if converged:
            match_name, idx = match_stored(final, stored_patterns, pattern_names)
            if match_name is not None:
                print(f"✓ Converged to stored pattern {match_name} (index {idx})")
            else:
                print("✓ Converged to a pattern (not in stored set)")
        else:
            print("✗ Did not converge within max steps")

    print("\n" + "="*60)
    print("--- Noisy Pattern Recall Demo (with energy tracking) ---")
    print("="*60)
    
    for name, pattern in zip(pattern_names, stored_patterns):
        for flips in (0, 1, 2):
            noisy_pattern = flip_bits(pattern, flips, seed=42)
            print(f"\n{name} with {flips} bit flip(s):")
            print(f"  Noisy input: {noisy_pattern}")

            final, steps, converged, energy_hist = recall(
                noisy_pattern, W, activation, max_steps=20, track_energy=True
            )
            
            print(f"  Output: {final}")
            
            if energy_hist:
                initial_energy = energy_hist[0]
                final_energy = energy_hist[-1]
                print(f"  Energy: {initial_energy:.2f} → {final_energy:.2f}")
            
            match_name, idx = match_stored(final, stored_patterns, pattern_names)
            if match_name is not None:
                print(f"  ✓ Recalled: {match_name}")
            else:
                print(f"  ✗ Failed to recall a stored pattern")
    
    print("\n" + "="*60)



