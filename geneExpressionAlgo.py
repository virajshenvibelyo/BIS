import numpy as np

# Define the optimization problem (e.g., f(x, y) = x^2 + y^2)
def optimization_function(solution):
    return solution[0]*2 + solution[1]*2

# Initialize parameters
def initialize_parameters():
    population_size = 50
    num_genes = 2  # Dimensionality of the solution
    mutation_rate = 0.1
    crossover_rate = 0.8
    num_generations = 100
    return population_size, num_genes, mutation_rate, crossover_rate, num_generations

# Initialize population
def initialize_population(population_size, num_genes):
    return np.random.uniform(-10, 10, (population_size, num_genes))

# Evaluate fitness
def evaluate_fitness(population):
    return np.array([optimization_function(ind) for ind in population])

# Selection
def select_parents(population, fitness):
    probabilities = 1 / (fitness + 1e-6)  # Avoid division by zero
    probabilities /= probabilities.sum()
    indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[indices]

# Crossover
def crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents) and np.random.rand() < crossover_rate:
            point = np.random.randint(1, parents.shape[1])
            offspring1 = np.concatenate((parents[i, :point], parents[i + 1, point:]))
            offspring2 = np.concatenate((parents[i + 1, :point], parents[i, point:]))
            offspring.extend([offspring1, offspring2])
        else:
            offspring.extend([parents[i], parents[i + 1] if i + 1 < len(parents) else parents[i]])
    return np.array(offspring)

# Mutation
def mutate(offspring, mutation_rate):
    for individual in offspring:
        if np.random.rand() < mutation_rate:
            gene = np.random.randint(individual.size)
            individual[gene] += np.random.normal(0, 1)  # Gaussian mutation
    return offspring

# Gene expression (identity mapping in this example)
def gene_expression(population):
    return population

# Gene Expression Algorithm
def gene_expression_algorithm():
    # Step 1: Initialize parameters
    population_size, num_genes, mutation_rate, crossover_rate, num_generations = initialize_parameters()

    # Step 2: Initialize population
    population = initialize_population(population_size, num_genes)

    best_solution = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        # Step 3: Evaluate fitness
        fitness = evaluate_fitness(population)

        # Track the best solution
        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < best_fitness:
            best_fitness = fitness[min_fitness_idx]
            best_solution = population[min_fitness_idx]

        # Step 4: Selection
        parents = select_parents(population, fitness)

        # Step 5: Crossover
        offspring = crossover(parents, crossover_rate)

        # Step 6: Mutation
        population = mutate(offspring, mutation_rate)

        # Step 7: Gene Expression
        population = gene_expression(population)

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    print(f"Best Solution: {best_solution}, Best Fitness: {best_fitness}")
    return best_solution, best_fitness

# Run the algorithm
if __name__ == "__main__":
    gene_expression_algorithm()