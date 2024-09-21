import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import tqdm
import multiprocessing as mp
import dill
from multiprocessing import Pool

# Set multiprocessing to use dill
import multiprocessing.reduction
multiprocessing.reduction.ForkingPickler = dill

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# # evaluation
# def evaluate(env, x):
#     return np.array(list(map(lambda y: simulation(env,y), x)))

class Specialist():
    def __init__(self, n_hidden_neurons=10,
                 experiment_name = 'optimization_test_arco',
                 upperbound = 1,
                 lowerbound = -1,
                 population_size = 100,
                 ) -> None:
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
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        self.n_vars = (self.env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.population_size = population_size

    def simulation(self, neuron_values):
        f, p, e, t = self.env.play(pcont=neuron_values)
        return f

    def fitness_eval(self, population):
        return np.array([simulation(self.env, individual) for individual in population])
    
    
    # def fitness_eval(self, population):
    #     with Pool(mp.cpu_count()) as pool:
    #         fitness_population = pool.starmap(self.simulation, [(self.env, individual) for individual in population])
    #     return np.array(fitness_population)

    def initialize(self):
        return np.random.uniform(self.lowerbound,
                                self.upperbound,
                                (self.population_size, self.n_vars))

    def mutation(self, child, p_mutation=0.2):

        if np.random.uniform() > p_mutation:
            #no mutation
            return child
        else:
            child = np.array(child)
            swap1, swap2 = np.random.choice(np.arange(1,10), size=2)
            child[[swap1, swap2]] = child[[swap2, swap1]]
            child_mutated = list(child)
        return child_mutated

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

            n_offspring = np.random.randint(1,3+1, 1)[0]
            offspring = np.zeros( (n_offspring, self.n_vars) )

            for f in range(0,n_offspring):

                cross_prop = np.random.uniform(0,1)
                offspring[f] = p1*cross_prop+p2*(1-cross_prop)

                # mutation
                for i in range(0,len(offspring[f])):
                    if np.random.uniform(0 ,1)<=p_mutation:
                        offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

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
        print(chosen)
        chosen = np.append(chosen[1:],np.argmax(new_fitness_population))
        print(chosen)
        print(chosen.shape)
        pop = new_population[chosen]
        fit_pop = new_fitness_population[chosen]

        return pop, fit_pop


    def train(self, total_generations=100):
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
        for gen_idx in tqdm.tqdm(range(generation_number, total_generations)):
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

if __name__ == '__main__':
    Specialist(experiment_name = 'optimization_test_arco2', population_size=50).train(total_generations=100)
