"""
wrap mlrose optmization algorithms with fixed sets of "common" hyper parameters to be used across problems
for comparing performance across algorithms
"""

import mlrose2 as mlrose
import numpy as np
import pandas as pd
import time
import random



def GA(problem):
    # Solve problem using the genetic algorithm
    best_state, best_fitness, eval_loss, current_best_loss = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=50, 
                                                            max_iters=np.inf)
    return best_state, best_fitness, eval_loss, current_best_loss


def RHC(problem):
    # Solve problem using the genetic algorithm
    best_state, best_fitness, eval_loss, current_best_loss = mlrose.random_hill_climb(problem, max_attempts=10, 
                                                                  max_iters=np.inf, restarts=0, init_state=None)
    return best_state, best_fitness, eval_loss, current_best_loss

def RHC_restart(problem):
    # Solve problem using the genetic algorithm
    best_state, best_fitness, eval_loss, current_best_loss = mlrose.random_hill_climb(problem, max_attempts=10, 
                                                                  max_iters=np.inf, restarts=5, init_state=None)
    return best_state, best_fitness, eval_loss, current_best_loss


def SA(problem):
    # Solve problem using the genetic algorithm
    best_state, best_fitness, eval_loss, current_best_loss = mlrose.simulated_annealing(problem, max_attempts=10, 
                                                                    schedule=mlrose.GeomDecay(init_temp=10000, decay=0.95, min_temp=0.01),
                                                                    max_iters=np.inf, init_state=None)
    return best_state, best_fitness, eval_loss, current_best_loss


def MIMIC(problem): 
    # Solve problem using the genetic algorithm
    best_state, best_fitness, eval_loss, current_best_loss = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=np.inf)
    
    return best_state, best_fitness, eval_loss, current_best_loss


def batch_optimize(problems, algorithm):
    start = time.time()
    best_fitnesses = []
    num_iterations = []
    
    for p in problems:
        if algorithm == 'RHC':
            best_state, best_fitness, eval_loss, current_best_loss = RHC(p)
        elif algorithm == 'RHC_restart':
            best_state, best_fitness, eval_loss, current_best_loss = RHC_restart(p)
        elif algorithm == 'GA':
            best_state, best_fitness, eval_loss, current_best_loss = GA(p)
        elif algorithm == 'SA':
            best_state, best_fitness, eval_loss, current_best_loss = SA(p)
        elif algorithm == 'MIMIC':
            best_state, best_fitness, eval_loss, current_best_loss = MIMIC(p)
        else:
            raise Exception("""algorithm does not exist.""")
            
        best_fitnesses.append(best_fitness)
        num_iterations.append(len(eval_loss))
    
    end = time.time()
    time_seconds = end - start 
    avg_best_fitness = sum(best_fitnesses)/len(best_fitnesses)
    avg_iterations = sum(num_iterations)/len(num_iterations)

    return time_seconds, avg_best_fitness, avg_iterations, best_fitnesses, num_iterations   

