# EVOLVE-BLOCK-START
"""Adaptive Genetic Algorithm for Circle Packing of n=26 Circles"""

import numpy as np

def initialize_population(pop_size, n):
    """Initialize a population of circle packs with random configurations."""
    population = []
    for _ in range(pop_size):
        centers = np.random.rand(n, 2) * 0.98 + 0.01  # Random centers within bounds
        radii = np.ones(n) * (1.0 / n)  # Start with equal radii approximating to 1/n
        population.append((centers, radii))
    return population

def fitness(centers, radii):
    """Calculate the fitness of a circle packing configuration."""
    # Calculate the sum of radii which we want to maximize
    total_radii = np.sum(radii)
    return total_radii

def mutate(centers, radii):
    """Randomly mutate a circle packing to explore new configurations."""
    mutation_strength = 0.02
    mutation_type = np.random.choice(['position', 'size'])
    if mutation_type == 'position':
        idx = np.random.randint(len(centers))
        centers[idx] += np.random.rand(2) * mutation_strength - mutation_strength / 2  # Small random move
    else:
        idx = np.random.randint(len(radii))
        radii[idx] *= (1 + (np.random.rand() - 0.5) * 0.1)  # Small random size change
    return np.clip(centers, 0.01, 0.99), np.clip(radii, 0.01, None)

def crossover(parent1, parent2):
    """Perform crossover between two parents to create two offspring."""
    centers1, radii1 = parent1
    centers2, radii2 = parent2
    crossover_point = np.random.randint(len(centers1))

    # Create two offspring
    offspring1_centers = np.vstack((centers1[:crossover_point], centers2[crossover_point:]))
    offspring2_centers = np.vstack((centers2[:crossover_point], centers1[crossover_point:]))
    offspring1_radii = np.concatenate((radii1[:crossover_point], radii2[crossover_point:]))
    offspring2_radii = np.concatenate((radii2[:crossover_point], radii1[crossover_point:]))

    return (offspring1_centers, offspring1_radii), (offspring2_centers, offspring2_radii)

def select_parents(population):
    """Select two parents for reproduction based on their fitness."""
    weights = np.array([fitness(centers, radii) for centers, radii in population])
    return np.random.choice(population, size=2, p=weights/weights.sum())

def evolve_population(population, pop_size):
    """Evolve the circle packings over several generations."""
    new_population = []
    for _ in range(pop_size // 2):
        parent1, parent2 = select_parents(population)
        offspring1, offspring2 = crossover(parent1, parent2)
        # Mutate both offspring
        offspring1 = mutate(*offspring1)
        offspring2 = mutate(*offspring2)
        new_population.append(offspring1)
        new_population.append(offspring2)
    return new_population

def run_evolution(num_generations=100, pop_size=50, n=26):
    """Run the genetic algorithm to evolve the best circle packing."""
    population = initialize_population(pop_size, n)

    for generation in range(num_generations):
        population = evolve_population(population, pop_size)

    # Select the best solution found
    best_centers, best_radii = max(population, key=lambda x: fitness(x[0], x[1]))
    return best_centers, best_radii

def construct_packing():
    """Construct a specific arrangement of 26 circles in a unit square."""
    return run_evolution()

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
