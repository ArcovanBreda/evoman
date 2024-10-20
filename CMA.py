import argparse
from evoman.environment import Environment
from demo_controller import player_controller
from parent_selection import dynamic_selection
import cma

import numpy as np
import os
import tqdm


class Generalist():
    def __init__(self) -> None:
        self.parse_args()

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.controller = player_controller(self.n_hidden_neurons)

        self.best_fitness = 0
        self.env = Environment(experiment_name=self.experiment_name,
                enemies=self.enemy_train,
                playermode="ai",
                player_controller=self.controller,
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False,
                multiplemode="yes", 
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

    def _vis_weights_fitness(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3))
        labels = ["Player energy", "Enemy energy", "Time"]
        colors = ["purple", "orange", "cyan"]

        # plot weights for each term in fitness function
        # for i in range(self.wfg.shape[0]):
            # plt.plot(np.array(range(self.total_generations)) + 1, self.wfg[i], label=labels[i], alpha=0.5)
        plt.plot(np.array(range(self.total_generations)) + 1, self.wfg[1], label=labels[1], color=colors[1], alpha=0.4, linewidth=3)
        plt.plot(np.array(range(self.total_generations)) + 1, self.wfg[0], label=labels[0], color=colors[0], alpha=0.4, linewidth=3)
        plt.plot(np.array(range(self.total_generations)) + 1, self.wfg[2], label=labels[2], color=colors[2], alpha=1)

        # plot settings
        plt.title(f'Term Weights across Generations with ε={self.fitness_epsilon:.2f}', fontsize=14)
        plt.xlabel('Generations', fontsize=14)
        plt.ylabel('Weights', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0, 1)
        plt.xlim(1, self.total_generations)
        plt.grid()
        plt.legend(fontsize=12, bbox_to_anchor=(1, 0.75))
        plt.tight_layout()

        # save plot
        os.makedirs(self.experiment_name, exist_ok=True)
        plt.savefig(self.experiment_name + "/weight_distribution.png")

    def _weights_fitness_gradual(self, gens, epsilon, num_terms=3):
        """
           Calculates self.wfg (num_terms, self.total_generations), containing
           the parameter values per generation for dynamic_fitness_gradual.
           wfg[0, :] is Enemy Energy
           wfg[1, :] is Player Energy
           wfg[2, :] is time

           Args:
            num_terms - int, number of weights in fitness function
            gens - list of length 4, contains the generation numbers to hold values constant over
                   and to gradually change them over
            epsilon - float, lowest probability
        """
        # init weights for each generation
        self.wfg = np.zeros((num_terms, self.total_generations))

        # init for first gens
        self.wfg[1, :gens[0]] = 1 - epsilon
        self.wfg[0, :gens[0]] = epsilon / (num_terms - 1)

        # gradual transition
        steps_1 = np.linspace(1 - epsilon, epsilon / (num_terms - 1), gens[1] - gens[0])
        self.wfg[1, gens[0]:gens[1]] = steps_1
        self.wfg[0, gens[0]:gens[1]] = np.flip(steps_1)

        # hold constant
        self.wfg[1, gens[1]:gens[2]] = np.array([steps_1[-1]] * (gens[2] - gens[1]))
        self.wfg[0, gens[1]:gens[2]] = np.array([steps_1[0]] * (gens[2] - gens[1]))

        # time term is constant until penultimate gens
        self.wfg[2, :gens[2]] = epsilon / (num_terms - 1)

         # move them to equal weight
        end_weight = 1 / num_terms
        self.wfg[0, gens[2]:gens[3]] = np.linspace(self.wfg[0, gens[2] - 1], end_weight, gens[3] - gens[2])
        self.wfg[1, gens[2]:gens[3]] = np.linspace(self.wfg[1, gens[2] - 1], end_weight, gens[3] - gens[2])
        self.wfg[2, gens[2]:gens[3]] = np.linspace(self.wfg[2, gens[2] - 1], end_weight, gens[3] - gens[2])

        # keep constant for final gens
        self.wfg[0, gens[3]:] = np.array([self.wfg[0, gens[3]-1]] * (self.total_generations - gens[3]))
        self.wfg[1, gens[3]:] = np.array([self.wfg[1, gens[3]-1]] * (self.total_generations - gens[3]))
        self.wfg[2, gens[3]:] = np.array([self.wfg[2, gens[3]-1]] * (self.total_generations - gens[3]))

        # visualise weight probability distribution
        self._vis_weights_fitness()

    def fitness_eval_gradual(self, population):
        # remove step sizes from individuals when using controller
        if self.mutation_type == "uncorrelated":
            _, params = population.shape
            population = population[:, :params - self.mutation_stepsize]

        fs, ps, es, ts = self.get_stats(population)
        enemies_selected = np.array(self.enemy_train)-1

        # get all information
        static_fitness = self.get_mean(fs, enemies_selected)
        defeated = es[:, np.array(self.enemy_train)-1]
        enemies_defeated_number = [len([x for x in defeat if x <= 0]) for defeat in defeated]

        ps = np.array(self.get_mean(ps, enemies_selected))
        es = np.array(self.get_mean(es, enemies_selected))
        ts = np.array(self.get_mean(ts, enemies_selected))

        max_enemies_defeated = np.max(enemies_defeated_number)
        enemies_defeated = enemies_defeated_number.count(max_enemies_defeated)
        print(f"{enemies_defeated} individuals have defeated {np.max(enemies_defeated_number)} enemies")

        # max_enemies_defeated = np.max(enemies_defeated_total)
        # enemies_defeated = enemies_defeated_total.count(max_enemies_defeated)
        # enemies_defeated_total = np.array(enemies_defeated_total)

        # # print number of defeated enemies
        # print(f"{enemies_defeated} individuals have defeated  {max_enemies_defeated} enemies")

        # custom fitness function
        fitness = self.wfg[0, self.generation_number] * ps # player health
        + self.wfg[1, self.generation_number] * (100 - es) # enemy health, '100 - es' is same as in baseline fitness func
        - self.wfg[2, self.generation_number] * np.log(np.minimum(ts, 3000)) # log time is same as in baseline fitness func

        return np.array([list(item) for item in zip(fitness, ps, es, ts, enemies_defeated_number, static_fitness)])

    def fitness_eval_stepwise(self, population):
        # remove step sizes from individuals when using controller
        if self.mutation_type == "uncorrelated":
            _, params = population.shape
            population = population[:, :params - self.mutation_stepsize]

        fs, ps, es, ts = self.get_stats(population)
        enemies_selected = np.array(self.enemy_train)-1

        # get all information
        # static_fitness = self.get_mean(fs, [0, 1, 2, 3, 4, 5, 6, 7])
        static_fitness = self.get_mean(fs, enemies_selected)
        defeated = es[:, np.array(self.enemy_train)-1]

        # per individual list with defeated enemies
        enemies_defeated_total = np.array([[1 if x <= 0 else 0 for x in defeat] for defeat in defeated])
        enemies_defeated_number = [len([x for x in defeat if x <= 0]) for defeat in defeated]

        ps = np.array(self.get_mean(ps, enemies_selected))
        es = np.array(self.get_mean(es, enemies_selected))
        ts = np.array(self.get_mean(ts, enemies_selected))

        # print number of defeated enemies
        max_enemies_defeated = np.max(enemies_defeated_number)
        enemies_defeated = enemies_defeated_number.count(max_enemies_defeated)
        print(f"{enemies_defeated} individuals have defeated {np.max(enemies_defeated_number)} enemies")
        # print(f"{np.max(enemies_defeated_number)} individuals have defeated  {np.max(enemies_defeated_number)} enemies")

        enemies_score = np.array([16, 18, 40, 50, 20, 12, 23, 22])  # e1': 16, 'e2': 18, 'e3': 40, 'e4': 50, 'e5': 20, 'e6': 12, 'e7': 23, 'e8': 22,
        enemies = [en - 1 for en in self.enemy_train]
        enemies_score = enemies_score[enemies]
        fitness = np.array(np.sum(enemies_defeated_total * enemies_score, axis=-1)) + 0.8135347605703016 * (100 - es) + 0.1 * ps - np.log(np.minimum(ts, 3000))

        return np.array([list(item) for item in zip(fitness, ps, es, ts, enemies_defeated_number, static_fitness)])

    def initialize(self):
        data = []
        with open(f"experiments/run{5}_popusize=100_enemy=[1, 2, 3, 4, 5, 6, 7, 8]_gens=1000_hiddensize=10_u=1_l=-1_fitnessfunc=steps_init=kaiming_seed={5}/best.txt") as f:
            for line in f:
                data.append(float(line))
        data = np.array(data)
        return data
        # total_weights = np.tile(data, (self.population_size, 1)) + np.random.randn(self.population_size, 265) * 0.01

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
            # total_weights[0, :] = data
        else:
            total_weights = np.random.uniform(self.lowerbound, self.upperbound, (self.population_size, self.n_vars - self.mutation_stepsize))

        if self.mutation_type == 'uncorrelated':
            sigmas = np.ones((self.population_size, self.mutation_stepsize)) * self.s_init

            return np.hstack((total_weights, sigmas))

        return total_weights

    def train(self):
        # if no earlier training is done:
        if not os.path.exists(self.experiment_name+'/results.txt'):
            print("hallo")
            guess = self.initialize()
            print(guess)
            # guess = [0]*265
            self.generation_number = 0
        else:
            print(f"Found earlier state for: {self.experiment_name}")
            self.env.load_state()
            guess = []
            with open(self.experiment_name + '/best.txt', 'r') as f:
                for l in f:
                    guess.append(float(l))

            # find generation we left of at:
            with open(self.experiment_name + '/results.txt', 'r') as f:
                for line in f:
                    l = line
                self.generation_number = int(l.strip().split()[1]) + 1

            if self.generation_number >= self.total_generations:
                print("\n\nAlready fully trained\n\n")
                return

        es = cma.CMAEvolutionStrategy(guess, self.s_init, {'popsize': self.population_size, 'seed': self.seed})

        for gen_idx in tqdm.tqdm(range(self.generation_number, self.total_generations)):
            self.generation_number = gen_idx
            population = np.array(es.ask())
            fitness_results = self.fitness_func(population)
            # fitness, ps, es, ts, enemies_defeated_number, static_fitness
            static_fitness, fitness = fitness_results[:, -1], fitness_results[:, 0]
            p, e = fitness_results[:, 1], fitness_results[:, 2]
            # es.tell(list(population), list(-(p - e))) # MINUS BECAUSE CMA DOES MINIMALISATION
            es.tell(list(population), list(-fitness)) # MINUS BECAUSE CMA DOES MINIMALISATION
            # es.tell(list(population), list(-(static_fitness))) # MINUS BECAUSE CMA DOES MINIMALISATION

            best_idx = np.argmax(fitness)
            print(f"(RUN {self.experiment_name.split('/')[-1][0:5]}): Static: {static_fitness[best_idx]:.4f}, dynamic: {fitness[best_idx]:.4f}, static mean: {static_fitness.mean():.4f}")            
            
            with open(self.experiment_name + '/results.txt', 'a') as f:
                # save as best, mean, std
                # print(f"Generation {gen_idx}: {static_fitness[best_idx]:.5f} {mean:.5f} {std:.5f}" )
                f.write(f"Generation: {gen_idx} {static_fitness[best_idx]:.4f} {fitness[best_idx]:.4f} {static_fitness.mean():.4f} {static_fitness.std():.4f}\n")
            
            if fitness[best_idx] > self.best_fitness:
                print(f"\nNEW BEST: {fitness[best_idx]}\n")
                np.savetxt(self.experiment_name + '/best.txt', population[best_idx])
                self.best_fitness = fitness[best_idx]
            

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
        parser.add_argument('-lc', '--log_custom', action="store_true", help="Log info custom fitness function as well")
        parser.add_argument('-hi', '--handin', action="store_true", help="Slice sigma values of genes when storing best solution")
        parser.add_argument('-ff', '--fitness_function', default='steps', choices=['steps', 'gradual'], help='Choose a fitness function')
        parser.add_argument('-g', '--gens', type=str, default='20,40,60,80', help='4 generation numbers to adjust fitness function weights over')
        parser.add_argument('-fe', '--fitness_epsilon', type=float, default=0.2, help="Lowest probability for term in fitness function is fitness_epsilon / 2 ")

        parser.add_argument('-se', '--seed', type=int, default=1, help='seed')

        args = parser.parse_args()
        self.seed = args.seed
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
        self.log_custom = True if args.log_custom else False
        self.handin = True if args.handin else False

        # CMA-ES Parameters
        if self.mutation_type == 'correlated':
            self.c_sigma = args.c_sigma       # Learning rate for the step-size path
            self.c_c = args.c_c               # Learning rate for the covariance matrix path
            self.c_1 = args.c_1               # Learning rate for the covariance matrix update
            self.c_mu = args.c_mu             # Learning rate for the covariance matrix update
            self.d_sigma = args.d_sigma       # Damping for the step-size update
            self.sigma = args.sigma_init_corr # Initial step size

        # usage check
        # if self.mutation_type == 'uncorrelated' and self.mutation_stepsize < 1:
        #     parser.error("--mutation_stepsize must be >= 1 for uncorrelated mutation")

        # file name generation
        self.experiment_name = 'experiments/' + args.experiment_name
        self.experiment_name += f'_popusize={self.population_size}'
        self.experiment_name += f'_enemy={self.enemy_train}'
        self.experiment_name += f'_gens={self.total_generations}'
        self.experiment_name += f'_hiddensize={self.n_hidden_neurons}'
        self.experiment_name += f'_u={self.upperbound}'
        self.experiment_name += f'_l={self.lowerbound}'
        self.experiment_name += f'_fitnessfunc={args.fitness_function}' #TODO idk if this messes up plot code
        if self.kaiming:
            self.experiment_name += f'_init=kaiming'
        else:
            self.experiment_name += f'_init=random'
        self.experiment_name += f"_seed={self.seed}"
        if args.fitness_function == "steps":
            self.fitness_func = self.fitness_eval_stepwise
        elif args.fitness_function == "gradual":
            self.fitness_func = self.fitness_eval_gradual
            change_gens = [int(g) for g in args.gens.split(',')]

            if len(change_gens) != 4:
                raise ValueError("Please provide 4 generation numbers to change weights over")
            if not 0 < args.fitness_epsilon <= 1:
                raise ValueError("Please assign non-zero probability to args.fitness_epsilon")
            self.fitness_epsilon = args.fitness_epsilon
            self._weights_fitness_gradual(gens=change_gens, epsilon=args.fitness_epsilon)
            
        return
if __name__ == '__main__':
    specialist = Generalist()
