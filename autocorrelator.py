# ...existing code...
def build_weight_matrix(patterns: List[np.ndarray]) -> np.ndarray:
    """Builds a Hopfield-style weight matrix from bipolar patterns (zero diagonal)."""
    if not patterns:
        raise ValueError("patterns must be a non-empty list of numpy arrays")
    n = patterns[0].size
    W = np.zeros((n, n), dtype=float)
    for p in patterns:
        if p.size != n:
            raise ValueError("all patterns must have the same size")
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W
# ...existing code...
def activation(x: np.ndarray):
    """Bipolar step activation function. Accepts scalar or array and returns same kind."""
    xa = np.asarray(x)
    out = np.where(xa >= 0, 1, -1)
    # return scalar for scalar input, ndarray otherwise
    if out.shape == ():
        return int(out)
    return out
# ...existing code...
    if mode == "synchronous":
        for step in range(1, max_steps + 1):
            # use W @ state (weights times state) for net input
            new_state = activation_fn(W @ state)
            if np.array_equal(new_state, state):
                return new_state, step, True
            state = new_state
        return state, max_steps, False
# ...existing code...
    elif mode == "asynchronous":
        for step in range(1, max_steps + 1):
            prev = state.copy()
            # random order asynchronous updates (one full pass = one step)
            for i in rng.permutation(n):
                net = W[i, :] @ state
                # activation_fn may return scalar for scalar input
                state[i] = activation_fn(net)
            if np.array_equal(state, prev):
                return state, step, True
        return state, max_steps, False
# ...existing code...
def match_stored(state: np.ndarray, stored: List[np.ndarray], names: Optional[List[str]] = None) -> Tuple[Optional[str], int]:
    """Return (name, index) of matching stored pattern or (None, -1)."""
    for i, p in enumerate(stored):
        if np.array_equal(state, p):
            return (names[i] if names else None, i)
    return (None, -1)
# ...existing code...