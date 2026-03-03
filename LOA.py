import time
import numpy as np
import random


def LOA(lyrebirds, fname, lb, ub, max_iterations):
    num_lyrebirds, num_variables = lyrebirds.shape
    # Parameters
    crossover_rate = 0.8  # Crossover rate
    mutation_rate = 0.1  # Mutation rate
    sigma = 0.1  # Mutation standard deviation
    fitness = np.zeros(num_lyrebirds)  # Fitness values
    convergence = np.zeros((max_iterations, 1))
    ct = time.time()

    # Main loop
    for iteration in range(1, max_iterations + 1):
        # Evaluate fitness for each lyrebird
        for i in range(num_lyrebirds):
            fitness[i] = fname(lyrebirds[i])

        # Sort lyrebirds based on fitness
        sorted_indices = np.argsort(fitness)
        lyrebirds = lyrebirds[sorted_indices]

        # Selection: keep the top half of lyrebirds
        lyrebirds = lyrebirds[:num_lyrebirds // 2]

        # Crossover
        num_crossovers = round(crossover_rate * num_lyrebirds // 2)
        for _ in range(num_crossovers):
            # Select two parents randomly
            r1 = random.randint(0, len(lyrebirds) - 1)
            r2 = random.randint(0, len(lyrebirds) - 1)
            parent1 = lyrebirds[r1]
            parent2 = lyrebirds[r2]

            # Perform crossover (blend crossover)
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

            # Add children to the population
            lyrebirds = np.vstack((lyrebirds, child1, child2))

        # Mutation
        num_mutations = round(mutation_rate * len(lyrebirds))
        for t in range(num_mutations):
            # Select a lyrebird randomly
            lyrebird_index = random.randint(0, len(lyrebirds) - 1)
            lyrebird = lyrebirds[lyrebird_index]

            # Perform mutation (additive Gaussian mutation)
            mutation = sigma * np.random.randn(num_variables)

            # Ensure mutated lyrebird stays within bounds [lb, ub]
            mutated_lyrebird = lyrebird + mutation
            mutated_lyrebird = np.clip(mutated_lyrebird, lb, ub)

            # Replace the old lyrebird with the mutated one
            lyrebirds[lyrebird_index] = mutated_lyrebird[t]

        convergence[iteration - 1] = np.min(fitness)

        # Display best solution
        best_solution = lyrebirds[0]
        best_fitness = fitness[0]

    ct = time.time() - ct
    return best_fitness, convergence, best_solution, ct


