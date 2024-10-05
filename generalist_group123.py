import argparse
from evoman.environment import Environment
from demo_controller import player_controller
from parent_selection import dynamic_selection

import numpy as np
import os
import tqdm


class Generalist():
    def __init__(self) -> None:
        self.parse_args()

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.controller = player_controller(self.n_hidden_neurons)

        self.env = Environment(experiment_name=self.experiment_name,
                enemies=[int(en) for en in self.enemy_train.split(",")],
                playermode="ai",
                player_controller=self.controller,
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False,
                multiplemode="yes", 
                randomini="yes", 
                # logs="yes"
                )
                
        self.global_env = Environment(experiment_name=self.experiment_name,
                enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                playermode="ai",
                player_controller=self.controller,
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False,
                multiplemode="yes", randomini="no")
        
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
        elif self.testing:
            self.test()

    def _get_name(self):
        return self.experiment_name

    def simulation(self, neuron_values):
        """Always eval with all enemies for correct logging."""
        f, p, e, t = self.global_env.play(pcont=neuron_values)
        return f, p, e, t

    def individual_gain(self, neuron_values):
        f, p, e, t = self.env.play(pcont=neuron_values)
        return p - e

    def fitness_eval(self, population):
        # remove step sizes from individuals when using controller
        print("Using: default fitness_eval") #TODO

        if self.mutation_type == "uncorrelated":
            _, params = population.shape
            population = population[:, :params - self.mutation_stepsize]

        fitness, _, _, _ = np.array([self.simulation(individual) for individual in population])
        return fitness, None #TODO zijn er wel 2 return waardes nodig?
    
    def dynamic_fitness_rules(self, population):
        print("Using: dynamic_fitness_rules") #TODO
        # remove step sizes from individuals when using controller
        if self.mutation_type == "uncorrelated":
            _, params = population.shape
            population = population[:, :params - self.mutation_stepsize]
        
        #TODO hier al return waardes opvangen van simulation en bewerken?
        #TODO 2 waardes returnen, de f van self.simulation als eerste en f_custom als tweede
        fitness_original, ps, es, ts = np.array([self.simulation(individual) for individual in population])
        fitness_custom = None

        return fitness_original, fitness_custom  #TODO zijn er wel 2 return waardes nodig?

    def dynamic_fitness_gradual(self, population):
        print("Using: dynamic_fitness_gradual") #TODO
        # remove step sizes from individuals when using controller
        if self.mutation_type == "uncorrelated":
            _, params = population.shape
            population = population[:, :params - self.mutation_stepsize]
        #TODO of hier de dynamische fitness funcs doen dus niet direct die self.simulation(individual zo storen, maar hier dan de fitness berekenen?)
        #TODO 2 waardes returnen, de f van self.simulation als eerste en f_custom als tweede
        fitness_original, ps, es, ts = np.array([self.simulation(individual) for individual in population])
        fitness_custom = None

        return fitness_original, fitness_custom #TODO zijn er wel 2 return waardes nodig?

    def initialize(self):
        if self.kaiming:
            n_actions = 5
            n_inputs = 20
            bias1 = np.zeros((self.population_size, self.n_hidden_neurons))
            bias2 = np.zeros((self.population_size, n_actions))

            limit1 = np.sqrt(1 / float(n_inputs))
            limit2 = np.sqrt(2 / float(self.n_hidden_neurons))
            weights1 = np.random.normal(0.0, limit1, size=(self.population_size, n_inputs * self.n_hidden_neurons))
            weights2 = np.random.normal(0.0, limit2, size=(self.population_size, self.n_hidden_neurons * n_actions))

            total_weights = np.hstack((bias1, weights1, bias2, weights2))
        else:
            total_weights = np.random.uniform(self.lowerbound, self.upperbound, (self.population_size, self.n_vars - self.mutation_stepsize))

        if self.mutation_type == 'uncorrelated':
            sigmas = np.ones((self.population_size, self.mutation_stepsize)) * self.s_init

            return np.hstack((total_weights, sigmas))

        return total_weights

    def mutation(self, child):
        """Expects a single child as input and mutates it."""
        if np.random.uniform() > self.mutation_probability:
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
                raise NotImplementedError("Please set mutation step size to 1 or equal to # parameters")
        elif self.mutation_type == 'correlated':
            # CMA-ES mutation
            child_mutated = np.random.multivariate_normal(mean=child, cov=self.sigma**2 * self.C)
        elif self.mutation_type == 'addition':
            # to check old behavior
            child_mutated = [i + np.random.normal(0, 1) if np.random.uniform() <= self.mutation_probability else i for i in range(len(child))]

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

    def selection(self, new_population, new_fitness_population):
        # TODO: REWRITE
        fitness = np.clip(new_fitness_population, 1e-10, None)
        probs = (fitness)/np.sum(fitness)
        chosen = np.random.choice(new_population.shape[0], self.population_size , p=probs, replace=False)
        pop = new_population[chosen]
        fit_pop = new_fitness_population[chosen]

        return pop, fit_pop

    def train(self):
        # if no earlier training is done:
        if not os.path.exists(self.experiment_name+'/results.txt'):
            # See if there is a run with 100 runs before continuing with 200
            experiment_name2 = self.experiment_name.split("_")
            experiment_name2[3] = "gens=100"
            experiment_name2 = "_".join(experiment_name2)
            if not os.path.exists(experiment_name2+'/results.txt'):

                population = self.initialize()
                generation_number = 0
            else:
                self.env = Environment(experiment_name=experiment_name2,
                    enemies=[self.enemy_train],
                    playermode="ai",
                    player_controller=self.controller,
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
                self.env.load_state()
                population = self.env.solutions[0]
                print(f"Found earlier state for: {self.experiment_name}")

                self.env = Environment(experiment_name=self.experiment_name,
                    enemies=[self.enemy_train],
                    playermode="ai",
                    player_controller=self.controller,
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

                # find generation we left of at:
                with open(experiment_name2 + '/results.txt', 'r') as f:
                    for line in f:
                        l = line
                    generation_number = int(l.strip().split()[1][:-1]) + 1

                if generation_number >= self.total_generations:
                    print("\n\nAlready fully trained\n\n")
                    return

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

        fitness_population, fitness_popu_custom = self.fitness_func(population) #TODO aanpassen hier
        # fitness_population= self.fitness_eval(population)[:, 0]
        #TODO call dyna_fit_pop = ...., so we should collect all of the terms...

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
            fitness_offspring, fitness_offspring_custom = self.fitness_func(offspring)
            # fitness_offspring = self.fitness_eval(offspring)[:, 0]
            #TODO dyna_fit_offspring = ...., so we should collect all of the terms... and probably replace it downwards or smth

            new_fitness_population = np.hstack((fitness_population, fitness_offspring))

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
        parser.add_argument('-etr', '--enemy_train', type=str, default="2", help='Enemy to fight during training')
        parser.add_argument('-ete', '--enemy_test', type=str, default='2', help='Enemy to fight during testing (1,3,5)')
        parser.add_argument('-t', '--train', action="store_true", help='Train EA')
        parser.add_argument('-is', '--intermediate_save', action="store_true", help="Don't automaticaly save states.")
        parser.add_argument('-v', '--visualise_best', action="store_true", help="Shows the character when testing")
        parser.add_argument('-mp', '--mutation_probability', type=float, default=0.5, help="probability an individual gets mutated")
        parser.add_argument('-te', '--test', action="store_true", help="Tests the selected bot / enemies")
        parser.add_argument('-ff', '--fitness_function', default='default', choices=['default', 'dyna_rules', 'dyna_gradual'])

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
        self.mutation_probability = args.mutation_probability
        self.testing = args.test

        if args.fitness_function == "default":
            self.fitness_func = self.fitness_eval
        elif args.fitness_function == "dyna_rules":
            self.fitness_func = self.dynamic_fitness_rules
        elif args.fitness_function == "dyna_gradual":
            self.fitness_func = self.dynamic_fitness_gradual

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
        self.experiment_name += f'_gens={self.total_generations}'
        self.experiment_name += f'_hiddensize={self.n_hidden_neurons}'
        self.experiment_name += f'_u={self.upperbound}'
        self.experiment_name += f'_l={self.lowerbound}'
        self.experiment_name += f'_mutationtype={self.mutation_type}'
        self.experiment_name += f'_mutationprobability={self.mutation_probability}'

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

    def test(self, type="individual gain"):
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
                scores.append(self.simulation(best_individual)[0])
                print(f"Fitness of the best individual against enemy {enemy}: {scores[-1]}")
            elif type == "individual gain":
                scores.append(self.individual_gain(best_individual))
                print(f"Fitness of the best individual against enemy {enemy}: {scores[-1]}")
            else:
                raise NotImplementedError
        return scores

if __name__ == '__main__':
    specialist = Generalist()
