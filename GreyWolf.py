import numpy as np

def gwo(obj_function, dim, search_agents, max_iter, lb, ub):
    # Initialize alpha, beta, and delta positions
    Alpha_pos = np.zeros(dim)
    Beta_pos = np.zeros(dim)
    Delta_pos = np.zeros(dim)

    Alpha_score = float("inf")
    Beta_score = float("inf")
    Delta_score = float("inf")

    # Initialize the positions of search agents
    positions = np.random.uniform(lb, ub, (search_agents, dim))

    for iteration in range(max_iter):
        for i in range(search_agents):
            # Constrain positions within search space
            positions[i] = np.clip(positions[i], lb, ub)
            
            # Evaluate the fitness of each agent
            fitness = obj_function(positions[i])
            
            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score, Alpha_pos = fitness, positions[i].copy()
            elif fitness < Beta_score:
                Beta_score, Beta_pos = fitness, positions[i].copy()
            elif fitness < Delta_score:
                Delta_score, Delta_pos = fitness, positions[i].copy()

        # Print the current best score at each iteration
        print(f"Iteration {iteration + 1}/{max_iter}, Best Score: {Alpha_score:.6f}")

        # Update the position of each search agent
        a = 2 - iteration * (2 / max_iter)  # Linearly decreases from 2 to 0

        for i in range(search_agents):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()

                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3  # Average of Alpha, Beta, Delta

    return Alpha_pos, Alpha_score

# Example: Optimization of the Sphere function
def sphere_function(x):
    return np.sum(x**2)

# Parameters
dim = 5                    # Dimensionality
search_agents = 30         # Number of wolves
max_iter = 50           # Maximum iterations
lb, ub = -10, 10           # Search space boundaries

best_position, best_score = gwo(sphere_function, dim, search_agents, max_iter, lb, ub)
print("Best Position:", best_position)
print("Best Score:", best_score)


