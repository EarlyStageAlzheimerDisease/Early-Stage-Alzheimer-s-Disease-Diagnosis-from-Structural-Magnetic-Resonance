import random
import time
import numpy as np


def DMO(Positions, fobj, VRmin, VRmax, Max_iter):
    # Initialization
    N, dim = Positions.shape
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    # Parameters
    nBabysitter = 3
    nAlphaGroup = N - nBabysitter
    nScout = nAlphaGroup
    L = round(0.6 * dim * nBabysitter)
    peep = 2

    # Variables
    Convergence_curve = np.zeros((Max_iter, 1))
    BestSol = None
    BEF = float('inf')  # Best fitness
    BEP = None  # Best position
    C = np.zeros(N)  # Counter for each mongoose
    sm = np.zeros(N)  # Fitness change rate
    tau = float('inf')  # Average fitness change
    pop = Positions.copy()

    # Initial fitness evaluation
    fitness = np.array([fobj(pop[i, :]) for i in range(N)])
    BEF = np.min(fitness)
    BEP = pop[np.argmin(fitness), :].copy()

    # Main optimization loop
    start_time = time.time()
    for iter in range(Max_iter):
        CF = (1 - iter / Max_iter) ** (2 * iter / Max_iter)  # Convergence factor

        # Calculate fitness proportional selection probabilities
        MeanCost = np.mean(fitness)
        F = np.exp(-fitness / MeanCost)
        P = F / np.sum(F)

        for i in range(N):
            # Update positions (Alpha Group and Scouts)
            k = random.choice([j for j in range(N) if j != i])
            phi = (peep / 2) * random.uniform(-1, 1)
            new_position = pop[i] + phi * (pop[i] - pop[k])
            new_position = np.clip(new_position, lb, ub)
            new_cost = fobj(new_position)

            if new_cost < fitness[i]:
                pop[i] = new_position
                fitness[i] = new_cost
                C[i] = 0  # Reset counter if improved
            else:
                C[i] += 1  # Increment counter if no improvement

            sm[i] = (new_cost - fitness[i]) / max(new_cost, fitness[i])

        # Babysitter rule
        for i in range(nBabysitter):
            if C[i] >= L:
                pop[i] = np.random.uniform(lb, ub, dim)
                fitness[i] = fobj(pop[i])
                C[i] = 0

        # Update global best
        current_best = np.min(fitness)
        if current_best < BEF:
            BEF = current_best
            BEP = pop[np.argmin(fitness), :].copy()

        # Scout behavior based on sm and tau
        new_tau = np.mean(sm)
        for i in range(nScout):
            M = np.mean(pop, axis=0)  # Mean position
            phi = (peep / 2) * random.uniform(-1, 1)
            if new_tau > tau:
                new_position = pop[i] - CF * phi * np.random.rand(dim) * (pop[i] - M)
            else:
                new_position = pop[i] + CF * phi * np.random.rand(dim) * (pop[i] - M)
            new_position = np.clip(new_position, lb, ub)
            new_cost = fobj(new_position)

            if new_cost < fitness[i]:
                pop[i] = new_position
                fitness[i] = new_cost

        tau = new_tau
        Convergence_curve[iter] = BEF

    end_time = time.time()

    return BEF, Convergence_curve, BEP, end_time - start_time







