import sys
from evoman.environment import Environment
from demo_controller import player_controller
from parent_selection import dynamic_selection

import numpy as np
import os
import tqdm


class Specialist():
    def __init__(self, n_hidden_neurons=10,
                 experiment_name='optimization_test_arco',
                 upperbound=1,
                 lowerbound=-1,
                 population_size=100,
                 c_sigma=0.3,
                 c_c=0.2,
                 d_sigma=0.5,
                 c_1=0.1,
                 c_mu=0.1,
                 sigma_init=0.5) -> None:
        self.headless = True
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.experiment_name = experiment_name
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.n_hidden_neurons = n_hidden_neurons
        self.env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        self.n_vars = (self.env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.population_size = population_size

        # CMA-ES Parameters
        self.c_sigma = c_sigma  # Learning rate for the step-size path
        self.c_c = c_c          # Learning rate for the covariance matrix path
        self.c_1 = c_1          # Learning rate for the covariance matrix update
        self.c_mu = c_mu        # Learning rate for the covariance matrix update
        self.d_sigma = d_sigma  # Damping for the step-size update
        self.sigma = sigma_init # Initial step size

        # CMA-ES State
        self.p_sigma = np.zeros(self.n_vars)
        self.p_c = np.zeros(self.n_vars)
        self.C = np.eye(self.n_vars)
        self.m = []

    def simulation(self, neuron_values):
        # print(neuron_values.shape)
        f, p, e, t = self.env.play(pcont=neuron_values)
        return f

    def fitness_eval(self, population):
        return np.array([self.simulation(individual) for individual in population])

    def initialize(self):
        return np.random.uniform(self.lowerbound, self.upperbound, (self.population_size, self.n_vars))

    def mutation(self, child):
        # CMA-ES mutation
        if np.random.uniform() > 0.6:
            return child

        mutated_child = np.random.multivariate_normal(mean=child, cov=self.sigma**2 * self.C)
        mutated_child = np.clip(mutated_child, self.lowerbound, self.upperbound)
        return mutated_child

    def update_evolution_paths(self, m, m_prime, population):
        #  Tracks how far the search is moving and helps adjust the step size for balanced exploration
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma)) * (m - m_prime) / self.sigma
        # Tracks the direction of the search and helps update the covariance matrix, improving the ability to mutate correlated parameters
        self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2 - self.c_c)) * (m - m_prime) / self.sigma

        self.C = (1 - self.c_1 - self.c_mu) * self.C + self.c_1 * np.outer(self.p_c, self.p_c) + self.c_mu * np.sum(
            [np.outer(ind - m_prime, ind - m_prime) / self.sigma**2 for ind in population], axis=0)

        # Update step size sigma
        self.sigma = self.sigma * np.exp((np.linalg.norm(self.p_sigma) / np.sqrt(
            1 - (1 - self.c_sigma)**(2 * (self.generation_number + 1))) - 1) / self.d_sigma)

    def limits(self, x):
        return np.clip(x, self.lowerbound, self.upperbound)

    def tournament(self, pop):
        c1 = np.random.randint(0,pop.shape[0], 1)
        c2 = np.random.randint(0,pop.shape[0], 1)

        c1_fit = self.fitness_eval(pop[c1])
        c2_fit = self.fitness_eval(pop[c2])

        if c1_fit > c2_fit:
            return pop[c1][0]
        else:
            return pop[c2][0]

    # def crossover(self, pop, p_mutation):
    #     total_offspring = np.zeros((0, self.n_vars))
    #     for p in range(0, pop.shape[0], 2):
    #         p1 = self.tournament(pop)
    #         p2 = self.tournament(pop)

    #         n_offspring = np.random.randint(1, 4)
    #         offspring = np.zeros((n_offspring, self.n_vars))

    #         for f in range(n_offspring):
    #             cross_prop = np.random.uniform(0, 1)
    #             offspring[f] = p1 * cross_prop + p2 * (1 - cross_prop)
    #             offspring[f] = self.mutation(offspring[f])
    #             total_offspring = np.vstack((total_offspring, offspring[f]))

    #     return total_offspring
    
    def crossover(self, parents, p_mutation):
        total_offspring = np.zeros((0, self.n_vars))
        
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i+1]

            n_offspring = np.random.randint(1, 4)
            offspring = np.zeros((n_offspring, self.n_vars))

            for f in range(n_offspring):
                cross_prop = np.random.uniform(0, 1)
                offspring[f] = p1 * cross_prop + p2 * (1 - cross_prop)
                offspring[f] = self.mutation(offspring[f])
                total_offspring = np.vstack((total_offspring, offspring[f]))

        return total_offspring
    
    def normalize(self, pop, fit):
        if (max(fit) - min(fit)) > 0:
            x_norm = (pop - min(fit)) / (max(fit) - min(fit))
        else:
            x_norm = 0
        return max(x_norm, 0.0000000001)

    def selection(self, new_population, new_fitness_population):
        fit_pop_norm = np.array([self.normalize(y, new_fitness_population) for y in new_fitness_population])
        probs = fit_pop_norm / fit_pop_norm.sum()
        chosen = np.random.choice(new_population.shape[0], self.population_size, p=probs, replace=False)
        pop = new_population[chosen]
        fit_pop = new_fitness_population[chosen]
        return pop, fit_pop

    def train(self, total_generations=100):
        if not os.path.exists(self.experiment_name + '/evoman_solstate'):
            population = self.initialize()
            self.generation_number = 0
        else:
            print("Found earlier state")
            self.env.load_state()
            population = self.env.solutions[0]
            with open(self.experiment_name + '/results.txt', 'r') as f:
                for line in f:
                    l = line
                self.generation_number = int(l.strip().split()[1][:-1])

        fitness_population = self.fitness_eval(population)

        for gen_idx in tqdm.tqdm(range(self.generation_number, total_generations)):
            self.generation_number = gen_idx

            # create parents
            parents = dynamic_selection(population, fitness_population, self.generation_number+1)

            # create new population (consisting of offspring)
            new_population_offspring = self.crossover(parents, p_mutation=0.2)

            # fitness of new population
            fitness_population_offspring = self.fitness_eval(new_population_offspring)

            # select from offspring
            population, fitness_population = self.selection(new_population_offspring, fitness_population_offspring)

            print(population.shape[0])

            # # create offspring
            # offspring = self.crossover(population, p_mutation=0.2)# PLACEHOLDER
            # # mutated_offspring = [self.mutation(springie) for springie in offspring]

            # new_population = np.vstack((population, offspring))

            # # evaluate new population
            # new_fitness_population = self.fitness_eval(new_population)

            # # select population to continue to next generation
            # population, fitness_population = self.selection(new_population, new_fitness_population)

            # save metrics for post-hoc evaluation
            best = self.fitness_eval([population[np.argmax(fitness_population)]])[0]
            mean = np.mean(fitness_population)
            std  = np.std(fitness_population)

            with open(self.experiment_name + '/results.txt', 'a') as f:
                # save as best, mean, std
                print(f"Generation {gen_idx}: {best:.5f} {mean:.5f} {std:.5f}" )
                f.write(f"Generation {gen_idx}: {best:.5f} {mean:.5f} {std:.5f}\n")

            np.savetxt(self.experiment_name + '/best.txt', population[np.argmax(fitness_population)])
            solutions = [population, fitness_population]
            self.env.update_solutions(solutions)
            self.env.save_state()

            # add mean for updating
            self.m.append(np.mean(population, axis=0))
            mean = self.m[-1]
            prev_m = self.m[-2] if len(self.m) > 1 else mean

            # update for CMA
            self.update_evolution_paths(mean, prev_m, population)

if __name__ == '__main__':
    Specialist(experiment_name='optimization_test_arco5', population_size=200, n_hidden_neurons=10).train(total_generations=100)
