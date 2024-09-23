import argparse
from evoman.environment import Environment
from demo_controller import player_controller
from uncorrelated_controller import uncorrelated_controller
from parent_selection import dynamic_selection

import numpy as np
import os
import tqdm


class Specialist():
    def __init__(self) -> None:
        self.parse_args()

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        if self.mutation_type == 'uncorrelated':
            controller = uncorrelated_controller(self.n_hidden_neurons, self.mutation_stepsize)
        else:
            controller = player_controller(self.n_hidden_neurons)

        self.env = Environment(experiment_name=self.experiment_name,
                enemies=[self.enemy_train],
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

        if self.trainmode:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.train()

    def simulation(self, neuron_values):
        f, p, e, t = self.env.play(pcont=neuron_values)
        return f

    def individual_gain(self, neuron_values):
        f, p, e, t = self.env.play(pcont=neuron_values)
        return p - e

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

    def crossover(self, parents):
        total_offspring = np.zeros((0, self.n_vars))

        for i in range(len(parents)):
            p1 = parents[i]
            if i == len(parents) - 1:
                p2 = parents[0]
            else:
                p2 = parents[i+1]

            cross_prop = 0.5
            offspring = p1 * cross_prop + p2 * (1 - cross_prop)
            offspring = self.mutation(offspring)
            total_offspring = np.vstack((total_offspring, offspring))

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
        # TODO: REWRITE
        p = np.array([self.normalize(pop, new_fitness_population) for pop in range(len(new_population))])
        p = p / p.sum()
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
        if not os.path.exists(self.experiment_name+'/results.txt'):
            population = self.initialize()
            generation_number = 0

        else:
            print(f"Found earlier state for: {self.experiment_name}")
            self.env.load_state()
            population = self.env.solutions[0]

            # find generation we left of at:
            with open(self.experiment_name + '/results.txt', 'r') as f:
                for line in f:
                    l = line
                generation_number = int(l.strip().split()[1][:-1]) + 1
            
            if generation_number >= self.total_generations:
                print("\n\nAlready fully trained\n\n")
                return 

            if self.mutation_type == 'correlated':
                with np.load(self.experiment_name + '/CMA_params.npz') as CMA_params:
                    self.p_sigma = CMA_params['p_sigma']
                    self.p_c = CMA_params['p_c']
                    self.C = CMA_params['C']
                    self.m = list(CMA_params['m'])

        fitness_population = self.fitness_eval(population)

        # Evolution loop
        for gen_idx in tqdm.tqdm(range(generation_number, self.total_generations)):
            self.generation_number = gen_idx
            # create parents
            parents = dynamic_selection(population, fitness_population, self.generation_number+1)

            # create new population (consisting of offspring)
            offspring = self.crossover(parents)

            # new population
            new_population = np.vstack((population, offspring))

            # evaluate new population
            new_fitness_population = self.fitness_eval(new_population)

            # select
            population, fitness_population = self.selection(new_population, new_fitness_population)

            # save metrics for post-hoc evaluation
            best = np.argmax(fitness_population)
            mean = np.mean(fitness_population)
            std  = np.std(fitness_population)

            if not self.intermediate_save:
                with open(self.experiment_name + '/results.txt', 'a') as f:
                    # save as best, mean, std
                    print(f"Generation {gen_idx}: {fitness_population[best]:.5f} {mean:.5f} {std:.5f}" )
                    f.write(f"Generation {gen_idx}: {fitness_population[best]:.5f} {mean:.5f} {std:.5f}\n")

                np.savetxt(self.experiment_name + '/best.txt', population[best])

                if self.mutation_type == 'correlated':
                    np.savez(self.experiment_name + '/CMA_params.npz',
                            p_sigma=self.p_sigma, p_c=self.p_c, C=self.C, m=self.m)

            solutions = [population, fitness_population]
            self.env.update_solutions(solutions)
            self.env.save_state()

            if self.mutation_type == "correlated":
                # add mean for updating
                self.m.append(np.mean(population, axis=0))
                mean = self.m[-1]
                prev_m = self.m[-2] if len(self.m) > 1 else mean

                # update for CMA
                self.update_evolution_paths(mean, prev_m, population)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-exp', '--experiment_name', type=str, default='test', help='Name of experiment')
        parser.add_argument('-ps', '--population_size', type=int, default=100, help='Size of the population')
        parser.add_argument('-tg', '--total_generations', type=int, default=100, help='Number of generations to run for')
        parser.add_argument('-n', '--n_hidden_neurons', type=int, default=10, help='Hidden layer size')
        parser.add_argument('-u', '--upperbound', type=int, default=1)
        parser.add_argument('-l', '--lowerbound', type=int, default=-1)
        parser.add_argument('-k', '--kaiming', action="store_true", help='Use Kaiming initialization of NN weights')
        parser.add_argument('-m', '--mutation', default='uncorrelated', choices=['uncorrelated', 'correlated', 'addition'])
        parser.add_argument('-ms', '--mutation_stepsize', type=int, default=0)
        parser.add_argument('-mt', '--mutation_threshold', type=float, default=0.001, help='epsilon_0 for uncorrelated mutation')
        parser.add_argument('-s', '--sigma_init', type=float, default=0.5, help='Init value for sigma(s) added to genes')
        parser.add_argument('-sc', '--sigma_init_corr', type=float, default=0.5, help='Init value for sigma(s) for correlated')
        parser.add_argument('-c', '--c_sigma', type=float, default=0.3, help='Init value for c_sigma (correlated)')
        parser.add_argument('-cc', '--c_c', type=float, default=0.2, help='Init value for c_c (correlated)')
        parser.add_argument('-ds', '--d_sigma', type=float, default=0.5, help='Init value for d_sigma (correlated)')
        parser.add_argument('-c1', '--c_1', type=float, default=0.1, help='Init value for c_1 (correlated)')
        parser.add_argument('-cmu', '--c_mu', type=float, default=0.1, help='Init value for c_mu (correlated)')
        parser.add_argument('-etr', '--enemy_train', type=int, default=2, help='Enemy to fight during training')
        parser.add_argument('-ete', '--enemy_test', type=str, default='2', help='Enemy to fight during testing (1,3,5)')
        parser.add_argument('-t', '--train', action="store_true", help='Train EA')
        parser.add_argument('-is', '--intermediate_save', action="store_true", help="Don't automaticaly save states.")
        parser.add_argument('-v', '--visualise_best', action="store_true", help="Shows the character when testing")

        
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
        self.enemy_train = args.enemy_train
        self.enemy_test = [int(x) for x in args.enemy_test.split(',')]
        self.trainmode = args.train
        self.intermediate_save = args.intermediate_save
        self.visualise_best = args.visualise_best

        # CMA-ES Parameters
        if self.mutation_type == 'correlated':
            self.c_sigma = args.c_sigma       # Learning rate for the step-size path
            self.c_c = args.c_c               # Learning rate for the covariance matrix path
            self.c_1 = args.c_1               # Learning rate for the covariance matrix update
            self.c_mu = args.c_mu             # Learning rate for the covariance matrix update
            self.d_sigma = args.d_sigma       # Damping for the step-size update
            self.sigma = args.sigma_init_corr # Initial step size

        # usage check
        if self.mutation_type == 'uncorrelated' and self.mutation_stepsize < 1:
            parser.error("--mutation_stepsize must be >= 1 for uncorrelated mutation")

        # file name generation
        self.experiment_name = 'experiments/' + args.experiment_name
        self.experiment_name += f'_popusize={self.population_size}'
        self.experiment_name += f'_enemy={self.enemy_train}'

        # if self.trainmode:
        #     self.experiment_name += '_mode=train'
        # else:
        #     self.experiment_name += '_mode=test'

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

    def test(self, type="fitness"):
        self.env.enemies = [self.enemy_test]
        if self.visualise_best:
            self.env.visuals = True 
            self.env.speed = "normal" 
            self.env.headless = False
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        best_individual_path = os.path.join(self.experiment_name, 'best.txt')
        if not os.path.exists(best_individual_path):
            raise FileNotFoundError(f"\n\nBest individual not found at {best_individual_path}, please set flag -t to train")

        best_individual = np.loadtxt(best_individual_path)

        # Simulate the environment with the best individual
        scores = []
        for enemy in self.enemy_test:
            self.env.enemies = [enemy]
            if type == "fitness":
                scores.append(self.simulation(best_individual))
                print(f"Fitness of the best individual against enemy {enemy}: {scores[-1]}")
            elif type == "individual gain":
                scores.append(self.individual_gain(best_individual))
                print(f"Fitness of the best individual against enemy {enemy}: {scores[-1]}")
            else:
                raise NotImplementedError
        return scores

if __name__ == '__main__':
    specialist = Specialist()
