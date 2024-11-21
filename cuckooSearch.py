import numpy as np
from scipy.special import gamma

# Objective Function (for example, Sphere function)
def objective(x):
    return np.sum(x**2)

# Lévy flight function
def levy_flight(beta, dim):
    sigma = (gamma(1+beta)*np.sin(np.pi*beta/2) / 
             (gamma((1+beta)/2)*beta*np.power(2, (beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / np.abs(v)**(1/beta)

# Cuckoo Search algorithm
def cuckoo_search(obj_func, dim, bounds, N=20, pa=0.25, max_iter=100):
    nests = np.random.uniform(bounds[0], bounds[1], (N, dim))
    fitness = np.array([obj_func(nest) for nest in nests])
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    for _ in range(max_iter):
        # Generate new solutions using Lévy flights
        new_nests = np.copy(nests)
        for i in range(N):
            step = levy_flight(1.5, dim)  # Lévy exponent 1.5
            new_nests[i] = nests[i] + 0.01 * step
            new_nests[i] = np.clip(new_nests[i], bounds[0], bounds[1])  # Bound the new nest position
        
        # Evaluate new solutions
        new_fitness = np.array([obj_func(nest) for nest in new_nests])
        
        # Abandon worst nests and replace with new ones
        for i in range(N):
            if np.random.rand() < pa and new_fitness[i] < fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]
        
        # Update the best solution
        best_nest_idx = np.argmin(fitness)
        best_nest = nests[best_nest_idx]
        best_fitness = fitness[best_nest_idx]
    
    return best_nest, best_fitness

# Parameters
dim = 10
bounds = [-5, 5]
best_nest, best_fitness = cuckoo_search(objective, dim, bounds)

print(f"Best Nest: {best_nest}")
print(f"Best Fitness: {best_fitness}")
