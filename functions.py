def transition_matrix(transitions):
    # Number of states (max = 1, n = 2)
    n = 1 + max(transitions)

    # Generate matrix M
    M = [[0]*n for _ in range(n)]

    for (i, j) in zip(transitions, transitions[1:]):
        M[i][j] += 1

    # Convert to probabilities
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M