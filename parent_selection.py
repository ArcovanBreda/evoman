import numpy as np


def roulette_wheel_selection(population, fitnesses):
    selected_individuals = []
    selected_fitness = []
    n = population.shape[0] 

    total_fitness = np.sum(fitnesses)

    for _ in range(int(n)): 
        alpha = np.random.uniform(0, total_fitness)
        cumulative_sum = 0
        j = 0

        # selection
        while cumulative_sum < alpha and j < n:
            cumulative_sum += fitnesses[j]
            j += 1

        selected_individuals.append(population[j-1])
        selected_fitness.append(fitnesses[j-1])

    return np.array(selected_individuals), np.array(selected_fitness)


def linear_rank_selection(population, fitnesses):
    selected_individuals = []
    selected_fitness = []
    n = population.shape[0] 

    sorted_indices = np.argsort(fitnesses)[::-1]
    ranks = np.zeros_like(sorted_indices)

    for rank, index in enumerate(sorted_indices, start=1):
        ranks[index] = rank

    probs = ranks / (n * (n - 1))

    value = 1 / (n - 2.001)

    while len(selected_individuals) < int(n):
        for i in range(n):
            alpha = np.random.uniform(0, value)
            for j in range(n):
                if probs[j] <= alpha:
                    if len(selected_individuals) < int(n):
                        selected_individuals.append(population[j])
                        selected_fitness.append(fitnesses[j])
                    break

    return np.array(selected_individuals), np.array(selected_fitness)


def exponential_rank_selection(population, fitnesses):
    selected_individuals = []
    selected_fitness = []
    n = population.shape[0] 

    sorted_indices = np.argsort(fitnesses)[::-1]
    ranks = np.zeros_like(sorted_indices)

    for rank, index in enumerate(sorted_indices, start=1):
        ranks[index] = rank

    probs = np.zeros(n)
    c = (n * 2 * (n - 1)) / (6 * (n - 1) + n)
    for i in range(n):
        probs[i] = 1.0 * np.exp( - ranks[i] / c)

    for _ in range(int(n)):
        alpha = np.random.uniform(1 / 9 * c, 2 / c)
        for j in range(n):
            if probs[j] <= alpha:
                selected_individuals.append(population[j])
                selected_fitness.append(fitnesses[j])
                break

    return np.array(selected_individuals), np.array(selected_fitness)


def tournament_selection(population, fitnesses):
    selected_individuals = []
    selected_fitness = []
    n = population.shape[0] 

    k = 30
    for _ in range(int(n)):
        temp = list(zip(population, fitnesses))
        np.random.shuffle(temp)
        res1, res2 = zip(*temp)
        shuffled_population, shuffled_fitnesses = np.array(res1), np.array(res2)

        # compare k individuals
        best_out_of_k = np.argmax(shuffled_fitnesses[0:k])
        selected_individuals.append(shuffled_population[best_out_of_k])
        selected_fitness.append(shuffled_fitnesses[best_out_of_k])

    return np.array(selected_individuals), np.array(selected_fitness)


def selection_score(population, fitness_population, generation):
    '''
    Selection based on the dynamic approach from
    'Parent Selection Operators for Genetic Algorithms'
    Input: current population, current generation number, fitness of current population
    Output: selection criterion
    '''
    best_idx = np.argmax(fitness_population)
    best = population[best_idx]
    criteria1 = 0
    pop_size = population.shape[0]

    # Hamming distance is binary so we use Manhattan distance instead
    for individual in population:
        criteria1 += np.sum(np.abs(best - individual))
    criteria1 /=  pop_size  # normalize
    criteria1 = np.exp(- criteria1 / generation)  # decrease over generations

    max_fitness = np.max(fitness_population)
    min_fitness = np.min(fitness_population)
    criteria2 = max_fitness / (max_fitness **2 + min_fitness**2)  # maximisation problem

    criterion = 1/generation * criteria1 + ((generation-1)/generation) * criteria2

    return criterion


def dynamic_selection(population, fitnesses, generation):

    rws_population, rws_fitness = roulette_wheel_selection(population, fitnesses)
    rws_score = selection_score(rws_population, rws_fitness, generation)

    lrs_population, lrs_fitness = linear_rank_selection(population, fitnesses)
    lrs_score = selection_score(lrs_population, lrs_fitness, generation)

    ers_population, ers_fitness = exponential_rank_selection(population, fitnesses)
    ers_score = selection_score(ers_population, ers_fitness, generation)

    tos_population, tos_fitness = tournament_selection(population, fitnesses)
    tos_score = selection_score(tos_population, tos_fitness, generation)

    scores = [rws_score, lrs_score, ers_score, tos_score]
    new_populations = [rws_population, lrs_population, ers_population, tos_population]
    best = np.argmax(scores)

    shuffled_population = np.random.permutation(new_populations[best])

    return shuffled_population
