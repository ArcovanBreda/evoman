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
                enemies=self.enemy_train,
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

        self.fitness_history = {'defeated': [], 'ig': []}
        self.fitness_weights = [20, 1, 0.1, 0.1]
        
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
        if self.mutation_type == "uncorrelated":
            _, params = population.shape
            population = population[:, :params - self.mutation_stepsize]

        return np.array([self.simulation(individual) for individual in population])

    def individual_simulation(self, neuron_values):
        
        fs, ps, es, ts = [], [], [], []

        # compute for all enemies
        for enemy in range(1, 9):
            
            self.env.enemies = [enemy]
            self.env.multiplemode = "no"
            f, p, e, t = self.env.play(pcont=neuron_values)
            
            fs.append(f), ps.append(p), es.append(e), ts.append(t)

        return fs, ps, es, ts

    def get_stats(self, population):
        fs, ps, es, ts = zip(*[self.individual_simulation(individual) for individual in population])

        return np.array(fs), np.array(ps), np.array(es), np.array(ts) 

    def get_mean(self, stats, indices):
        selected_elements = [[stat[i] for i in indices] for stat in stats]
        return np.mean(selected_elements, axis=1)
    
    def fitness_eval_stepwise(self, population):
        # remove step sizes from individuals when using controller
        if self.mutation_type == "uncorrelated":
            _, params = population.shape
            population = population[:, :params - self.mutation_stepsize]

        fs, ps, es, ts = self.get_stats(population)
        enemies_selected = np.array(self.enemy_train)-1

        static_fitness = self.get_mean(fs, [0, 1, 2, 3, 4, 5, 6, 7])
        defeated = es[:, np.array(self.enemy_train)-1]
        enemies_defeated_total = [len([x for x in defeat if x <= 0]) for defeat in defeated]
        ps = np.array(self.get_mean(ps, enemies_selected))
        es = np.array(self.get_mean(es, enemies_selected))
        ts = np.array(self.get_mean(ts, enemies_selected))

        max_enemies_defeated = np.max(enemies_defeated_total)
        enemies_defeated = enemies_defeated_total.count(max_enemies_defeated)
        enemies_defeated_total = np.array(enemies_defeated_total)
        # if max_enemies_defeated > int(len(self.enemy_train) / 2):
        #     enemies_defeated = len([x for x in enemies_defeated_total if x > int(len(self.enemy_train) / 2)])
        # elif max_enemies_defeated > 2:
        #     enemies_defeated = len([x for x in enemies_defeated_total if x > 2])
        # else: 
        #     enemies_defeated = enemies_defeated_total.count(max_enemies_defeated)
        
        # set back to original
        print(max_enemies_defeated, enemies_defeated)

        # only switch every n iters to avoid fluctuating too much
        iters = 5

        ig = ps - es
       
        if (self.generation_number+1) % iters == 0: # time to find new fitness function
            if all(x >= -10 for x in self.fitness_history['ig']) and all(x >= 4 for x in self.fitness_history['defeated']): # try to push more enemies
                self.fitness_weights[0] = 25
                self.fitness_weights[1] = 2
                self.fitness_weights[2] = 2
                self.fitness_weights[3] = 2
                print('PUSH ENEMIES')
            elif all(x >= -10 for x in self.fitness_history['ig']) and all(x >= 2 for x in self.fitness_history['defeated']):
                self.fitness_weights[0] = 20
                self.fitness_weights[1] = 1.5
                self.fitness_weights[2] = 2
                self.fitness_weights[3] = 2
                print('LOW IG')
            elif all(x >= -20 for x in self.fitness_history['ig']) and all(x >= 2 for x in self.fitness_history['defeated']):
                self.fitness_weights[0] = 15
                self.fitness_weights[1] = 1.5
                self.fitness_weights[2] = 2
                self.fitness_weights[3] = 2
                print('IG MET') # if baseline is met and ig is bigger than -20 (empirical tests), defeating and ig equally important, the rest becomes important
            elif all(x >= 2 for x in self.fitness_history['defeated']):
                self.fitness_weights[0] = 20
                self.fitness_weights[1] = 2
                self.fitness_weights[2] = 0.1
                self.fitness_weights[3] = 0.1
                print('BASELINE MET') # establish a baseline of defeating 2 enemies, and then prioritize ig
            else:
                self.fitness_weights[0] = 20
                self.fitness_weights[1] = 1
                self.fitness_weights[2] = 0.1
                self.fitness_weights[3] = 0.1
            self.fitness_history['defeated'] = []
            self.fitness_history['ig'] = []
        
        fitness = self.fitness_weights[0] * enemies_defeated_total + self.fitness_weights[1]*ig + self.fitness_weights[2] * ps - self.fitness_weights[3] * np.log(np.minimum(ts, 3000))

        self.fitness_history['defeated'].append(max_enemies_defeated)
        self.fitness_history['ig'].append(np.max(ig))

        print(self.fitness_history)
        print(self.fitness_weights)
        # enemies defeated and individual gain equally important after baseline has been established
        # fitness = 15 * enemies_defeated_total + 1.5*(ps - es) + 0.1 * ps - 0.1 * np.log(np.mininum(ts, 3000))

        # first set, need to establish baseline
        # if baseline, equal importance enemies and IG
        # if still baseline, and IG is at least



        '''
        if max_enemies_defeated == 0: # really bad focus on defeating enemies
            fitness = 100 - np.array(es)
            print('step terrible')
        elif np.min(np.array(es)) > 50: # for the best individual the enemies get half health
            fitness = 100 - np.array(es) + np.array(enemies_defeated_total)*10
            print('Nobody defeated enemies on average')
        elif np.min(np.array(es)) > 20:
            fitness = (100 - np.array(es)) + np.array(enemies_defeated_total)*20
            print('>20')
        elif np.min(np.array(es)) > 0: # for no individual defeating enemies on average is close
            fitness = 100 - np.array(es) + np.array(enemies_defeated_total)*5
            print('Somebody is close to defeating enemies')
        elif np.max(np.array(ps) - np.array(es)) >= 0: # best player does not lose on average
            fitness = np.array(ps) + (100 - np.array(es)) + np.array(enemies_defeated_total)*10
            print('step victory by best')
        else:
            fitness = np.array(ps) + 100 - np.array(es)
            print('step else')
        print("Individual Gain Max: ", np.max(np.array(ps) - np.array(es)))
        print("Player Health Max: ", np.max(np.array(ps)))
        print("Enemy Health Min: ", np.min(np.array(es)))
        '''

        '''
        if max_enemies_defeated == len(self.enemy_train):# and enemies_defeated >= int(0.25 * len(population)): # all enemies are defeated
            fitness = 0.1 * (100 - es) + 0.9 * np.array(ps) - np.log(np.minimum(np.array(ts), 3000))
            print('step all defeated')
        elif max_enemies_defeated > int(len(self.enemy_train) / 2):# and enemies_defeated >= int(0.25 * len(population)): # more than half of the enemies are defeated but not all
            fitness = 0.9 * np.array(ps) - 0.1 * (100 - np.array(es)) # individual gain #(enemies_defeated / len(self.env.enemies))**2 * (100 - np.array(e_e)) + (1 - (enemies_defeated / len(self.env.enemies))**2) * np.array(p_e)
            print('step more than half')
        elif max_enemies_defeated > 2:# and enemies_defeated >= int(0.25 * len(population)): # mostly focus on defeating enemies, moeite hier dus meer focussen op player
            fitness = np.array(ps) - np.array(es) #+ np.array(enemies_defeated_total)#0.8 * (100 - np.array(es)) + 0.2 * np.array(ps) + 0.8 * np.array(enemies_defeated_total) * 10
            print('step more than 2 less than half')
        else: # fully focus on defeating enemies if population that defeats is too small
            fitness = (100 - np.array(es)) + np.array(enemies_defeated_total) * 10
            print('step less than or equal to 2')
        '''


        # _, p_e, e_e, time = zip(*[self.simulation_local(individual) for individual in population])

        # # number of defeated enemies in de fitness function opnemen
        # fitness = np.zeros_like(static_fitness)
        # for i in range(len(population)):

        #     if enemies_defeated_total[i] == len(self.env.enemies): # all enemies are defeated
        #         fitness[i] = 0.1 * (100 - e_e[i]) + 0.9 * p_e[i] - np.log(np.minimum(time[i], 3000))
        #         # print('step all defeated')
        #     elif enemies_defeated_total[i] > int(len(self.env.enemies) / 2): # more than half of the enemies are defeated but not all
        #         fitness[i] = p_e[i] - e_e[i] # individual gain #(enemies_defeated / len(self.env.enemies))**2 * (100 - np.array(e_e)) + (1 - (enemies_defeated / len(self.env.enemies))**2) * np.array(p_e)
        #         # print('step more than half')
        #     elif enemies_defeated_total[i] >= 2: # mostly focus on defeating enemies
        #         fitness[i] = 0.8 * (100 - e_e[i]) + 0.2 * p_e[i] + 0.8 * enemies_defeated_total[i] * 10
        #         # print('step more than 2 less than half')
        #     else: # fully focus on defeating enemies if population that defeats is too small
        #         fitness[i] = (100 - e_e[i]) + enemies_defeated_total[i] * 10
        #         # print('step less than or equal to 2')

        # static_fitness = 0.9 * (100 - e_e) + 0.1 * p_e - log(time)

        # fitness = static_fitness

        return np.array([list(item) for item in zip(fitness, ps, es, ts, static_fitness)])


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

    def selection(self, new_population, new_fitness_population, new_static_fitness):
        # TODO: REWRITE
        fitness = np.clip(new_fitness_population, 1e-10, None)
        probs = (fitness)/np.sum(fitness)
        chosen = np.random.choice(new_population.shape[0], self.population_size , p=probs, replace=False)
        pop = new_population[chosen]
        fit_pop = new_fitness_population[chosen]
        stat_pop = new_static_fitness[chosen]

        return pop, fit_pop, stat_pop

    def train(self):
        # if no earlier training is done:
        if not os.path.exists(self.experiment_name+'/results.txt'):
            # See if there is a run with 100 runs before continuing with 200
            experiment_name2 = self.experiment_name.split("_")
            experiment_name2[3] = "gens=100"
            experiment_name2 = "_".join(experiment_name2)
            if not os.path.exists(experiment_name2+'/results.txt'):

                population = self.initialize()
                self.generation_number = 0
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
                    self.generation_number = int(l.strip().split()[1][:-1]) + 1

                if self.generation_number >= self.total_generations:
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
                self.generation_number = int(l.strip().split()[1][:-1]) + 1

            if self.generation_number >= self.total_generations:
                print("\n\nAlready fully trained\n\n")
                return

            if self.mutation_type == 'correlated':
                with np.load(self.experiment_name + '/CMA_params.npz') as CMA_params:
                    self.p_sigma = CMA_params['p_sigma']
                    self.p_c = CMA_params['p_c']
                    self.C = CMA_params['C']
                    self.m = list(CMA_params['m'])

        # fitness_population = self.fitness_eval_stepwise(population)[:, 0]
        fitness_results = self.fitness_eval_stepwise(population)
        fitness_population, static_fitness = fitness_results[:, 0], fitness_results[:, -1]
        # Evolution loop
        for gen_idx in tqdm.tqdm(range(self.generation_number, self.total_generations)):
            self.generation_number = gen_idx
            # create parents
            parents = dynamic_selection(population, fitness_population, self.generation_number+1)

            # create new population (consisting of offspring)
            offspring = self.crossover(parents)

            # new population
            new_population = np.vstack((population, offspring))

            # evaluate new population
            new_fitness_results = self.fitness_eval_stepwise(offspring)
            new_fitness_population = np.hstack((fitness_population, new_fitness_results[:, 0]))
            new_static_fitness = np.hstack((static_fitness, new_fitness_results[:, -1]))

            # select
            population, fitness_population, static_fitness = self.selection(new_population, new_fitness_population, new_static_fitness)

            # save metrics for post-hoc evaluation
            best = np.argmax(static_fitness)
            mean = np.mean(static_fitness)
            std  = np.std(static_fitness)

            if not self.intermediate_save:
                with open(self.experiment_name + '/results.txt', 'a') as f:
                    # save as best, mean, std
                    print(f"Generation {gen_idx}: {static_fitness[best]:.5f} {mean:.5f} {std:.5f}" )
                    f.write(f"Generation {gen_idx}: {static_fitness[best]:.5f} {mean:.5f} {std:.5f}\n")

                np.savetxt(self.experiment_name + '/best.txt', population[best])

                if self.mutation_type == 'correlated':
                    np.savez(self.experiment_name + '/CMA_params.npz',
                            p_sigma=self.p_sigma, p_c=self.p_c, C=self.C, m=self.m)

            solutions = [population, static_fitness]
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
        self.enemy_train = [int(en) for en in args.enemy_train.split(",")]
        self.enemy_test = [int(x) for x in args.enemy_test.split(',')]
        self.trainmode = args.train
        self.intermediate_save = args.intermediate_save
        self.visualise_best = args.visualise_best
        self.mutation_probability = args.mutation_probability
        self.testing = args.test

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
