import numpy as np
import time


# r = -t * ((-1) / maxiter) updated in line 22 & 27
def IDPOA(initsol, fitness_function, xmin, xmax, max_iter):
    Npop, Chlen = initsol.shape
    bestsol = None
    bestfit = float('inf')
    fitness = np.zeros(Npop)

    # Evaluate initial fitness
    for i in range(Npop):
        fitness[i] = fitness_function(initsol[i, :])
    bestfit = np.min(fitness)
    bestsol = initsol[np.argmin(fitness), :]

    start_time = time.time()

    for iteration in range(max_iter):
        for i in range(Npop):
            # Vaccination Phase
            random_factor = -iteration * ((-1) / max_iter)
            vaccinated_solution = initsol[i, :] + random_factor * (xmax[i, :] - xmin[i, :])
            vaccinated_solution = np.clip(vaccinated_solution, xmin[i, :], xmax[i, :])

            # Drug Administration Phase
            best_factor = -iteration * ((-1) / max_iter)
            drug_solution = initsol[i, :] + best_factor * (bestsol - initsol[i, :])
            drug_solution = np.clip(drug_solution, xmin[i, :], xmax[i, :])

            # Surgery Phase (for worst solutions)
            if fitness[i] > bestfit * 0.9:
                surgery_solution = 0.6 * initsol[i, :] + 0.4 * bestsol
                surgery_solution = np.clip(surgery_solution, xmin[i, :], xmax[i, :])
            else:
                surgery_solution = initsol[i, :]

            # Choose the best updated solution
            candidates = [vaccinated_solution, drug_solution, surgery_solution]
            candidate_fitness = [fitness_function(c) for c in candidates]
            best_candidate_index = np.argmin(candidate_fitness)

            # Update the solution if improved
            if candidate_fitness[best_candidate_index] < fitness[i]:
                initsol[i, :] = candidates[best_candidate_index]
                fitness[i] = candidate_fitness[best_candidate_index]

        # Update global best
        min_fitness_index = np.argmin(fitness)
        if fitness[min_fitness_index] < bestfit:
            bestfit = fitness[min_fitness_index]
            bestsol = initsol[min_fitness_index, :]

    end_time = time.time()
    return bestfit, fitness, bestsol, end_time - start_time
