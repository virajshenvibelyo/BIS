import numpy as np

# Define the optimization problem: Example is minimizing a quadratic function (e.g., f(x, y) = x^2 + y^2)
def optimization_function(position):
    return position[0]*2 + position[1]*2

# Initialize parameters
def initialize_parameters():
    grid_size = (10, 10)  # Grid size (10x10 grid)
    num_iterations = 100  # Number of iterations
    neighborhood_size = 1  # Size of the neighborhood (e.g., Moore neighborhood with radius 1)
    return grid_size, num_iterations, neighborhood_size

# Generate an initial population of cells with random positions
def initialize_population(grid_size):
    population = np.random.uniform(-10, 10, (grid_size[0], grid_size[1], 2))  # Random positions in 2D space
    return population

# Evaluate the fitness of each cell
def evaluate_fitness(population):
    fitness = np.zeros((population.shape[0], population.shape[1]))
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            fitness[i, j] = optimization_function(population[i, j])
    return fitness

# Update the state of each cell based on neighboring cells
def update_states(population, fitness, neighborhood_size):
    updated_population = np.copy(population)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            # Determine the neighborhood boundaries
            x_min = max(i - neighborhood_size, 0)
            x_max = min(i + neighborhood_size + 1, population.shape[0])
            y_min = max(j - neighborhood_size, 0)
            y_max = min(j + neighborhood_size + 1, population.shape[1])

            # Find the best neighbor (minimum fitness)
            best_neighbor = population[i, j]
            best_fitness = fitness[i, j]

            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    if fitness[x, y] < best_fitness:
                        best_neighbor = population[x, y]
                        best_fitness = fitness[x, y]

            # Update the cell's position toward the best neighbor
            updated_population[i, j] = (population[i, j] + best_neighbor) / 2

    return updated_population

# Main function to implement the Parallel Cellular Algorithm
def parallel_cellular_algorithm():
    # Step 1: Initialize parameters
    grid_size, num_iterations, neighborhood_size = initialize_parameters()

    # Step 2: Initialize population
    population = initialize_population(grid_size)

    # Step 3: Iterate over the algorithm
    best_solution = None
    best_fitness = float('inf')

    for iteration in range(num_iterations):
        # Step 4: Evaluate fitness
        fitness = evaluate_fitness(population)

        # Track the best solution
        min_fitness = np.min(fitness)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[np.unravel_index(np.argmin(fitness), fitness.shape)]

        # Step 5: Update states
        population = update_states(population, fitness, neighborhood_size)

        # Print progress
        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

    # Step 6: Output the best solution
    print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
    return best_solution, best_fitness

# Run the algorithm
if __name__ == "__main__":
    parallel_cellular_algorithm()