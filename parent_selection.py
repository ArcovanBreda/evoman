import numpy as np

# # dummy values for testing
# lowerbound = -1
# upperbound = 1
# population_size = 200
# n_vars = 265
# generation_num=100

# # population is (100, 265) with real values ranging from -1 to 1
# population = np.random.uniform(lowerbound, upperbound, (population_size, n_vars))
# fitness_population = np.random.uniform(50, 100, population_size)
# # end dummy values for testing



def roulette_wheel_selection(population, fitnesses):
    selected_individuals = []
    selected_fitness = []
    n = population.shape[0]

    total_fitness = np.sum(fitnesses)

    for _ in range(n): 
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


# Does not really work
# def stochastic_universal_sampling(population, fitnesses):
#     selected_individuals = []
#     selected_fitness = []
#     n = population.shape[0]

#     mean_fitness = np.mean(fitnesses)
#     alpha = np.random.rand()

#     cumulative_sum = fitnesses[0]
#     delta = alpha * mean_fitness

#     j = 0
#     while j < n - 1:
#         if delta < cumulative_sum:
#             selected_individuals.append(population[j])
#             selected_fitness.append(fitnesses[j])
#             delta += cumulative_sum
#             break
#         else:
#             j += 1
#             cumulative_sum += fitnesses[j] 

#     return np.array(selected_individuals), np.array(selected_fitness)




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

    while len(selected_individuals) < n:
        for i in range(n):
            alpha = np.random.uniform(0, value)
            for j in range(n):
                if probs[j] <= alpha:
                    if len(selected_individuals) < n:
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


    for i in range(n):
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

    k = 20
    for _ in range(n):
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

    # Hamming distance is binary so we use Euclidean distance instead
    for individual in population:
        criteria1 += np.linalg.norm(best - individual)
    criteria1 /=  pop_size # normalize
    criteria1 = np.exp(- criteria1 / generation) # decrease over generations

    max_fitness = np.max(fitness_population)
    min_fitness = np.min(fitness_population)
    criteria2 = max_fitness / (max_fitness **2 + min_fitness**2) # maximisation problem
    
    criterion = 1/generation * criteria1 + ((generation-1)/generation) * criteria2

    return criterion



def dynamic_selection(population, fitnesses, generation):

    rws_population, rws_fitness = roulette_wheel_selection(population, fitnesses)
    rws_score = selection_score(rws_population, rws_fitness, generation)

    # sus_population, sus_fitness = stochastic_universal_sampling(population, fitnesses)
    # sus_score = selection_score(sus_population, sus_fitness, generation)

    lrs_population, lrs_fitness = linear_rank_selection(population, fitnesses)
    lrs_score = selection_score(lrs_population, lrs_fitness, generation)

    ers_population, ers_fitness = exponential_rank_selection(population, fitnesses)
    ers_score = selection_score(ers_population, ers_fitness, generation)

    tos_population, tos_fitness = tournament_selection(population, fitnesses)
    tos_score = selection_score(tos_population, tos_fitness, generation)

    scores = [rws_score, lrs_score, ers_score, tos_score]
    new_populations = [rws_population, lrs_population, ers_population, tos_population]
    new_fitnesses = [rws_fitness, lrs_fitness, ers_fitness, tos_fitness]
    best = np.argmax(scores)
    # print(len(new_fitnesses[0]), len(new_fitnesses[1]), len(new_fitnesses[2]), len(new_fitnesses[3]))

    return new_populations[best], new_fitnesses[best]


# leave out truncation selection because it is not often used in practice and only for 
# very large populations

# for generation in range(1,100):
#     pop, pop_fit = dynamic_selection(population, fitness_population, generation)
#     print(generation, np.mean(pop_fit), np.std(pop_fit))
# print(parent_selection(population, generation_num, fitness_population))
# print(roulette_wheel_selection(population, fitness_population).shape)
# dynamic_selection(population, fitness_population, generation_num)

# print(population[0], population[1])