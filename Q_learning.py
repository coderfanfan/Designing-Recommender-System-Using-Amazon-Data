"""
Q-learning algorithm to find the optimal policies for reinforcement learning problem. 
Epsilon greedy is used to generate probabilities of each action to take next.
author: Yurong Fan (yurongfan@gmail.com)
reference:https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
"""
import gym
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from time import time


def epsilon_greedy_action(Q, observation, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
        observation: current state of the agent.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.   
    """
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A

def simulated_annealing_action(Q, observation, nA, init_temp, decay):
    """
    Creates simulated annealing policy based on a given Q-function and temperature settings.
    The algorithm is described in http://uhaweb.hartford.edu/compsci/ccli/projects/QLearning.pdf
    The probability of choosing an action given a state follows Boltzmann distribution. When temperature is high, all action will be selected, when temperature is low, optimal action based on Q-function has higher chance to be selected. 
    The temperatuer decrease over time following a geometric decay.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        init_temp: float, default: 1.0
            Initial value of temperature parameter T in a iteration. Must be greater than 0.
        decay: float, default: 0.9
            Temperature decay parameter, r. Must be between 0 and 1.
        nA: Number of actions in the environment.
        observation: current state of the agent.
    
    Returns:
        A: A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.   
        temp: the end temperature in this iteration after decay. 
    """
    temp = init_temp * decay
    prob_arr = Q[observation]/temp
    A = np.exp(prob_arr)/np.exp(prob_arr).sum() 
    return A, temp
    
def policy_from_Q(Q, nS, nA):    
    """
    Generate optimal policy from Q.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below).
        nA: Number of actions in the environment.
        nS: Number of states in the environment.
    
    Returns:
        The optimal policy that maps from state -> actions in which the best action has value 1 and the rest actions have value 0.
    """
    policy = np.zeros((nS, nA))
    for s in Q:
        best_action = np.argmax(Q[s])
        policy[s][best_action] = 1.0
    return policy


def value_from_Q(Q, nS):
    """
    Generate estimated value/utility of states from Q.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (number of actions in the environment).
        nS: Number of states in the environment.
    
    Returns:
        The estimated value/utility of states. 
    """
    V = np.empty(nS)
    V[:] = np.nan
    for s in Q:
        V[s] = max(Q[s])
    return V
 

def q_learning(env, num_episodes, next_action, discount_factor=1.0, alpha=0.5, epsilon=0.1, init_temp=1.0, decay=0.9999, max_iter=1000000):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        next_action: method to generate the next action. suported inputs are 'simulated_annealing', 'epsilon_greedy'.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1. only considered if next_action = 'epsilon_greedy'.
        init_temp: float, default: 1.0. 
            Initial value of temperature parameter T. Must be greater than 0. only considered if next_action = 'simulated_annealing'.
        decay: float, default: 0.9
            Temperature decay parameter, r. Must be between 0 and 1. only considered if next_action = 'simulated_annealing'.
        max_iter: number of maximum steps/actions to take in each episode 
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    nA = env.nA
    nS = env.nS
    Q = defaultdict(lambda: np.zeros(nA))
    
    # Capture statistics for analysis
    converge_iter = []
    time_iter = []
    start = time()
    reward_iter = []
    
    for i_episode in range(num_episodes):
        state = env.reset()
        eps_rew = 0
        for t in range(max_iter):           
            # Take a step
            if next_action == 'epsilon_greedy':
                action_probs = epsilon_greedy_action(Q, state, epsilon, nA)
            if next_action == 'simulated_annealing':
                action_probs, temp = simulated_annealing_action(Q, state, nA, init_temp, decay)
                init_temp = temp
            action = np.random.choice(np.arange(nA), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            eps_rew += reward

            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            # Update statistics
            converge_iter.append(td_delta)
            current = time()
            time_spt = (current - start) * 1000.0
            time_iter.append(time_spt)
            start = current
            
            if done:
                break
                
            state = next_state
        reward_iter.append(eps_rew)
            
    
    policy = policy_from_Q(Q, nS, nA)
    V = value_from_Q(Q, nS)
    
    return policy, V, converge_iter, time_iter, reward_iter