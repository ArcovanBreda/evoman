# imports framework
import sys, os

from evoman.environment import Environment

import numpy as np
from demo_controller import player_controller

# runs simulation
def simulation(env,x):
    # fitness, playerlife, enemylife, time
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test_hannah'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    

    # start writing your own code from here

    # Create population
        # Bounds for weights
        # Store everything in two dimensional array n_population x n_weights
        # Evaluate population (multiple of times) -> use network here

    # Parent Selection
        # Pairs of two so you can use for crossover
        # Rank from 0 but some stochasticity

    # Variation
        # Crossover
        # Mutation

    # Survivor Selection
        # Deterministically


    # Full algorithm
        # Parameters
        # population_size = 50
        # n_evaluations = 3
        # n_offspring = 50
        # weight_upper_bound = 2
        # weight_lower_bound = -2
        # mutation_sigma = .1
        # generations = 10

        # # Initialize environment, network and population. Perform an initial evaluation
        # env = gym.make("MountainCar-v0")
        # net = Perceptron (2, 3)
        # pop = initialize_population(population_size, weight_lower_bound, weight_upper_bound)
        # pop_fit = evaluate_population(pop, n_evaluations, net, env)

        # for i in range (generations):
        #     parents = parent_selection(pop, pop_fit, n_offspring)
        #     offspring = crossover (parents)
        #     offspring = mutate (offspring, weight_lower_bound, weight_upper_bound, mutation_sigma)

        #     offspring_fit = evaluate_population(offspring, n_evaluations, net, env)

        #     # concatenating to form a new population
        #     pop = np.vstack((pop,offspring))
        #     pop_fit = np.concatenate([pop_fit,offspring_fit])

        #     pop, pop_fit = survivor_selection(pop, pop_fit, population_size)

        #     print (f"Gen {i} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")
        #     clear_output(wait=True)
        # env.close()



if __name__ == '__main__':
    main()
