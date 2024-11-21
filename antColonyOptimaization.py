import random
import numpy as np
import math

# Define the Problem: Cities with coordinates (can be modified with real data)
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return math.sqrt((self.x - city.x)**2 + (self.y - city.y)**2)

# Ant Colony Optimization for TSP
class ACO_TSP:
    def __init__(self, cities, num_ants, num_iterations, alpha=1.0, beta=2.0, rho=0.5, q0=0.9):
        self.cities = cities
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Importance of pheromone
        self.beta = beta    # Importance of heuristic (distance)
        self.rho = rho      # Pheromone evaporation rate
        self.q0 = q0        # Probability of choosing the best path
        self.num_cities = len(cities)
        self.pheromone = np.ones((self.num_cities, self.num_cities))  # Initial pheromone values
        self.heuristic = np.zeros((self.num_cities, self.num_cities))  # Heuristic info (inverse of distance)
        self.best_tour = None
        self.best_tour_length = float('inf')

        # Compute heuristic (inverse of distance)
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = cities[i].distance(cities[j])
                self.heuristic[i][j] = 1.0 / dist if dist != 0 else 0
                self.heuristic[j][i] = self.heuristic[i][j]

    def select_next_city(self, current_city, visited_cities):
        probabilities = np.zeros(self.num_cities)
        total_pheromone = 0.0

        # Calculate the transition probabilities
        for city in range(self.num_cities):
            if city not in visited_cities:
                pheromone = self.pheromone[current_city][city] ** self.alpha
                heuristic = self.heuristic[current_city][city] ** self.beta
                probabilities[city] = pheromone * heuristic
                total_pheromone += probabilities[city]

        # Normalize probabilities
        if total_pheromone == 0:
            return random.choice([city for city in range(self.num_cities) if city not in visited_cities])

        probabilities /= total_pheromone

        # Exploration vs Exploitation
        if random.random() < self.q0:
            # Exploitation: choose the city with the highest probability
            next_city = np.argmax(probabilities)
        else:
            # Exploration: choose based on probabilities
            next_city = np.random.choice(self.num_cities, p=probabilities)

        return next_city

    def update_pheromone(self, ants):
        # Evaporate pheromone
        self.pheromone *= (1 - self.rho)

        # Deposit pheromone based on the ants' solutions
        for ant in ants:
            pheromone_deposit = 1.0 / ant.tour_length
            for i in range(self.num_cities):
                current_city = ant.tour[i]
                next_city = ant.tour[(i + 1) % self.num_cities]
                self.pheromone[current_city][next_city] += pheromone_deposit
                self.pheromone[next_city][current_city] += pheromone_deposit

    def run(self):
        for iteration in range(self.num_iterations):
            ants = [Ant(self.num_cities, self) for _ in range(self.num_ants)]
            for ant in ants:
                ant.construct_solution()

            # Update pheromones
            self.update_pheromone(ants)

            # Update the best solution found so far
            for ant in ants:
                if ant.tour_length < self.best_tour_length:
                    self.best_tour_length = ant.tour_length
                    self.best_tour = ant.tour

            print(f"Iteration {iteration + 1}/{self.num_iterations}: Best Tour Length = {self.best_tour_length}")

        return self.best_tour, self.best_tour_length

# Ant class to simulate each ant's behavior
class Ant:
    def __init__(self, num_cities, aco_tsp):
        self.num_cities = num_cities
        self.aco_tsp = aco_tsp
        self.tour = []
        self.tour_length = 0.0

    def construct_solution(self):
        start_city = random.randint(0, self.num_cities - 1)
        self.tour = [start_city]
        self.tour_length = 0.0
        visited_cities = set(self.tour)

        current_city = start_city
        while len(self.tour) < self.num_cities:
            next_city = self.aco_tsp.select_next_city(current_city, visited_cities)
            self.tour.append(next_city)
            visited_cities.add(next_city)
            self.tour_length += self.aco_tsp.cities[current_city].distance(self.aco_tsp.cities[next_city])
            current_city = next_city

        # Add the return to the starting city
        self.tour_length += self.aco_tsp.cities[self.tour[-1]].distance(self.aco_tsp.cities[self.tour[0]])

# Example usage

if __name__ == "__main__":
    # Define cities (x, y coordinates)
    cities = [City(0, 0), City(1, 3), City(4, 3), City(6, 1), City(3, 0)]
    
    # Initialize and run ACO
    aco = ACO_TSP(cities=cities, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, rho=0.5, q0=0.9)
    best_tour, best_tour_length = aco.run()
    
    # Output the best tour and its length
    print("\nBest tour found:", best_tour)
    print("Best tour length:", best_tour_length)
