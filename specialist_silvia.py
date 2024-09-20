import sys
import argparse

from evoman.environment import Environment
from demo_controller import player_controller
from uncorrelated_controller import uncorrelated_controller

# imports other libs
import numpy as np
import os
import tqdm


class Specialist():
    def __init__(self) -> None:
        self.parse_args()
    
        self.headless = True
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        controller = player_controller(self.n_hidden_neurons)

        if self.mutation_type == 'uncorrelated':
            controller = uncorrelated_controller(self.n_hidden_neurons, self.mutation_stepsize)
        
        self.env = Environment(experiment_name=self.experiment_name,
                    enemies=[self.enemy],
                    playermode="ai",
                    player_controller=controller,
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        self.n_vars = (self.env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5 + self.mutation_stepsize

        if self.mutation_type == 'correlated':
            # CMA-ES State
            self.p_sigma = np.zeros(self.n_vars)
            self.p_c = np.zeros(self.n_vars)
            self.C = np.eye(self.n_vars)
            self.m = []

    def simulation(self, neuron_values):
        f, p, e, t = self.env.play(pcont=neuron_values)
        return f


    def fitness_eval(self, population):
        return np.array([self.simulation(individual) for individual in population])


    def initialize(self):
        if self.kaiming:
            n_actions = 5
            n_inputs = 20
            bias1 = np.zeros((self.population_size, self.n_hidden_neurons))
            bias2 = np.zeros((self.population_size, n_actions))

            limit1 = np.sqrt(1 / float(n_inputs))
            limit2 = np.sqrt(2 / float(self.n_hidden_neurons))
            # print("Limits: ", limit1, limit2) #TODO remove later
            weights1 = np.random.normal(0.0, limit1, size=(self.population_size, n_inputs * self.n_hidden_neurons))
            weights2 = np.random.normal(0.0, limit2, size=(self.population_size, self.n_hidden_neurons * n_actions))

            total_weights = np.hstack((bias1, weights1, bias2, weights2))
        else:
            total_weights = np.random.uniform(self.lowerbound, self.upperbound, (self.population_size, self.n_vars - self.mutation_stepsize))
        
        if self.mutation_type == 'uncorrelated':
            sigmas = np.ones((self.population_size, self.mutation_stepsize)) * self.s_init

            return np.hstack((total_weights, sigmas))

        return total_weights


    def mutation(self, child, p_mutation=0.2):
        """Expects a single child as input and mutates it."""
        if np.random.uniform() > p_mutation:
            return child

        if self.mutation_type == 'uncorrelated':
            n = self.n_vars - self.mutation_stepsize

            if self.mutation_stepsize == 1:
                # update sigma
                tau = 1 / np.sqrt(n)
                sigma_prime = child[-1] * np.exp(np.random.normal(0, tau))
                sigma_prime = np.maximum(sigma_prime, self.mutation_threshold)

                # update child
                mutations = np.random.normal(0, sigma_prime, size=n)
                child_mutated = (child[:n] + mutations).tolist()
                child_mutated.append(sigma_prime)
            elif self.mutation_stepsize == n:
                # update sigmas
                tau_prime = 1 / np.sqrt(2 * n)
                tau = 1 / np.sqrt(2 * np.sqrt(n))
                sigma_prime = child[n:] * np.exp(np.random.normal(0, tau_prime) + np.random.normal(0, tau, size=n))
                sigma_prime = np.maximum(sigma_prime, self.mutation_threshold)

                # update child
                mutations = np.random.normal(0, sigma_prime)
                child_mutated = (child[:n] + mutations).tolist()
                child_mutated += sigma_prime.tolist()
            else:
                raise NotImplementedError #TODO everything for k, 1 < k < n should probably contain mapping from sigma to alleles 
        elif self.mutation_type == 'correlated':
            # CMA-ES mutation
            child_mutated = np.random.multivariate_normal(mean=child, cov=self.sigma**2 * self.C)
        elif self.mutation_type == 'addition':
            # to check old behavior
            child_mutated = [i + np.random.normal(0, 1) if np.random.uniform() <= p_mutation else i for i in range(len(child))]

        child_mutated = np.clip(child_mutated, self.lowerbound, self.upperbound)
        return child_mutated


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

        if x>self.upperbound:
            return self.upperbound
        elif x<self.lowerbound:
            return self.lowerbound
        else:
            return x


    def tournament(self, pop):
        c1 =  np.random.randint(0,pop.shape[0], 1)
        c2 =  np.random.randint(0,pop.shape[0], 1)

        c1_fit = self.fitness_eval(pop[c1])
        c2_fit = self.fitness_eval(pop[c2])

        if c1_fit > c2_fit:
            return pop[c1][0]
        else:
            return pop[c2][0]


    def crossover(self, pop, p_mutation):

        total_offspring = np.zeros((0,self.n_vars))

        for p in range(0, pop.shape[0], 2):
            p1 = self.tournament(pop)
            p2 = self.tournament(pop)

            n_offspring = np.random.randint(1, 3 + 1, 1)[0]
            offspring = np.zeros((n_offspring, self.n_vars))

            for f in range(0, n_offspring):

                cross_prop = np.random.uniform(0,1)
                offspring[f] = p1*cross_prop+p2*(1-cross_prop)

                # mutation
                offspring[f] = self.mutation(offspring[f], p_mutation)
                offspring[f] = np.array(list(map(lambda y: self.limits(y), offspring[f])))

                total_offspring = np.vstack((total_offspring, offspring[f]))

        return total_offspring


    def normalize(self, pop, fit):

        if (max(fit) - min(fit) ) > 0:
            x_norm = (pop - min(fit))/(max(fit) - min(fit))
        else:
            x_norm = 0
        if x_norm <= 0:
            x_norm = 0.0000000001
        return x_norm


    def selection(self, new_population, new_fitness_population):
        # TODO: find a better metric
        p = np.array([self.normalize(pop, new_fitness_population) for pop in range(len(new_population))])
        p = p / p.sum()
        # print(sorted(p))
        # chosen_idx = np.random.choice(np.arange(new_population.shape[0]),
        #                         size=self.population_size,
        #                         replace=False,
        #                         p=p)

        # chosen_idx = np.random.choice(np.arange(new_population.shape[0]),
        #                         size=self.population_size,
        #                         replace=True)
        # return new_population[chosen_idx, :], new_fitness_population[chosen_idx]

        fit_pop_cp = new_fitness_population
        fit_pop_norm =  np.array(list(map(lambda y: self.normalize(y,fit_pop_cp), new_fitness_population))) # avoiding negative probabilities, as fitness is ranges from negative numbers
        probs = (fit_pop_norm)/(fit_pop_norm).sum()
        chosen = np.random.choice(new_population.shape[0], self.population_size , p=probs, replace=False)
        # print(chosen)
        chosen = np.append(chosen[1:],np.argmax(new_fitness_population))
        # print("Chosen", chosen, chosen.shape)
        pop = new_population[chosen]
        fit_pop = new_fitness_population[chosen]

        return pop, fit_pop


    def train(self):
        # if no earlier training is done:
        if not os.path.exists(self.experiment_name+'/evoman_solstate'):
            population = self.initialize()
            generation_number = 0
        else:
            print("Found earlier state")
            self.env.load_state()
            population = self.env.solutions[0]

            # find generation we left off at:
            with open(self.experiment_name + '/results.txt', 'r') as f:
                for line in f:
                    l = line
                generation_number = int(l.strip().split()[1][:-1])

        # Log mean, best, std
        fitness_population = self.fitness_eval(population)
        # mean = np.mean(fitness_population)
        # best = np.argmax(fitness_population)
        # std = np.std(fitness_population)

        # Evolution loop
        for gen_idx in tqdm.tqdm(range(generation_number, self.total_generations)):
            # create offspring
            offspring = self.crossover(population, p_mutation=0.2)# PLACEHOLDER
            # mutated_offspring = [self.mutation(springie) for springie in offspring]

            new_population = np.vstack((population, offspring))

            # evaluate new population
            new_fitness_population = self.fitness_eval(new_population)

            # select population to continue to next generation
            population, fitness_population = self.selection(new_population, new_fitness_population)

            # save metrics for post-hoc evaluation
            best = np.argmax(fitness_population)
            mean = np.mean(fitness_population)
            std  = np.std(fitness_population)

            with open(self.experiment_name + '/results.txt', 'a') as f:
                # save as best, mean, std
                print(f"Generation {gen_idx}: {fitness_population[best]:.5f} {mean:.5f} {std:.5f}" )
                f.write(f"Generation {gen_idx}: {fitness_population[best]:.5f} {mean:.5f} {std:.5f}\n")

            np.savetxt(self.experiment_name + '/best.txt', population[best])
            solutions = [population, fitness_population]
            self.env.update_solutions(solutions)
            self.env.save_state()


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-exp', '--experiment_name', type=str, default='test', help='Name of experiment')
        parser.add_argument('-ps', '--population_size', type=int, default=100, help='Size of the population')
        parser.add_argument('-tg','--total_generations', type=int, default=100, help='Number of generations to run for')
        parser.add_argument('-n', '--n_hidden_neurons', type=int, default=10, help='Hidden layer size')
        parser.add_argument('-u', '--upperbound', type=int, default=1)
        parser.add_argument('-l', '--lowerbound', type=int, default=-1)
        parser.add_argument('-k', '--kaiming', action="store_true", help='Use Kaiming initialization of NN weights')
        parser.add_argument('-m', '--mutation', default='uncorrelated', choices=['uncorrelated', 'correlated', 'addition'])
        parser.add_argument('-ms', '--mutation_stepsize', type=int, default=0)
        parser.add_argument('-mt', '--mutation_threshold', type=float, default=0.01, help='epsilon_0 for uncorrelated mutation')
        parser.add_argument('-s', '--sigma_init', type=float, default=0, help='Init value for sigma(s) added to genes')
        parser.add_argument('-sc', '--sigma_init_corr', type=float, default=0.5, help='Init value for sigma(s) for correlated')
        parser.add_argument('-c', '--c_sigma', type=float, default=0.3, help='Init value for c_sigma (correlated)')
        parser.add_argument('-cc', '--c_c', type=float, default=0.2, help='Init value for c_c (correlated)')
        parser.add_argument('-ds', '--d_sigma', type=float, default=0.5, help='Init value for d_sigma (correlated)')
        parser.add_argument('-c1', '--c_1', type=float, default=0.1, help='Init value for c_1 (correlated)')
        parser.add_argument('-cmu', '--c_mu', type=float, default=0.1, help='Init value for c_mu (correlated)')
        parser.add_argument('-e', '--enemy', type=int, default=2, help='Enemy to fight')
        parser.add_argument('-t', '--train', action="store_true", help='Train EA')

        args = parser.parse_args()
        self.population_size = args.population_size
        self.total_generations = args.total_generations
        self.n_hidden_neurons = args.n_hidden_neurons
        self.upperbound = args.upperbound
        self.lowerbound = args.lowerbound
        self.kaiming = args.kaiming
        self.mutation_type = args.mutation
        self.mutation_stepsize = args.mutation_stepsize
        self.mutation_threshold = args.mutation_threshold
        self.s_init = args.sigma_init
        self.enemy = args.enemy
        self.trainmode = args.train

        # CMA-ES Parameters
        if self.mutation_type == 'correlated':
            self.c_sigma = args.c_sigma  # Learning rate for the step-size path
            self.c_c = args.c_c          # Learning rate for the covariance matrix path
            self.c_1 = args.c_1          # Learning rate for the covariance matrix update
            self.c_mu = args.c_mu        # Learning rate for the covariance matrix update
            self.d_sigma = args.d_sigma  # Damping for the step-size update
            self.sigma = args.sigma_init_corr # Initial step size

        # usage check
        if self.mutation_type == 'uncorrelated' and self.mutation_stepsize < 1:
            parser.error("--mutation_stepsize must be >= 1 for uncorrelated mutation")

        # file name generation
        self.experiment_name = 'experiments/' + args.experiment_name
        self.experiment_name += f'_popusize={self.population_size}'
        self.experiment_name += f'_enemy={self.enemy}'

        if self.trainmode:
            self.experiment_name += f'_mode=train'
        else:
            self.experiment_name += f'_mode=test'

        self.experiment_name += f'_gens={self.total_generations}'
        self.experiment_name += f'_hiddensize={self.n_hidden_neurons}'
        self.experiment_name += f'_u={self.upperbound}'
        self.experiment_name += f'_l={self.lowerbound}'
        self.experiment_name += f'_mutationtype={self.mutation_type}'

        if self.mutation_type == 'uncorrelated':
            self.experiment_name += f'_mutationstepsize={self.mutation_stepsize}'
            self.experiment_name += f'_mutationthreshold={self.mutation_threshold}'
            self.experiment_name += f'_sinit={self.s_init}'
        elif self.mutation_type == 'correlated':
            self.c_sigma = args.c_sigma  # Learning rate for the step-size path
            self.c_c = args.c_c          # Learning rate for the covariance matrix path
            self.c_1 = args.c_1          # Learning rate for the covariance matrix update
            self.c_mu = args.c_mu        # Learning rate for the covariance matrix update
            self.d_sigma = args.d_sigma  # Damping for the step-size update
            self.sigma = args.sigma_init_corr # Initial step size

            self.experiment_name += f'_csigma={self.c_sigma}'
            self.experiment_name += f'_cc={self.c_c}'
            self.experiment_name += f'_c1={self.c_1}'
            self.experiment_name += f'_cmu={self.c_mu}'
            self.experiment_name += f'_dsigma={self.d_sigma}'
            self.experiment_name += f'_sigma={self.sigma}'
            

        if self.kaiming:
            self.experiment_name += f'_init=kaiming'
        else:
            self.experiment_name += f'_init=random'

        return 


if __name__ == '__main__':
    specialist = Specialist()
    specialist.train()
