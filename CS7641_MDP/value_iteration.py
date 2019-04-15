"""
Value iteration algorithm to find the optimal policies for reinforcement learning problem
implemented by dennybritz and other contributors at 
https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
"""

import numpy as np
from time import time

def value_iteration(env, theta=0.0001, discount_factor=1.0, max_iter=100, early_stop=True):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        max_iter: maximum number of iterations. use with early_stop(see below) 
        early_stop: if set to true, stop if the convergence is reached before max_iter, otherwise run max_iter number of iterations. 
        
    Returns:
        A tuple (policy, converge, V, converge_iter, time_iter). 
        policy： is the last policy, a matrix of shape [S, A] where each state s contains a valid probability distribution over actions. 
        it's the optimal policy found by the algorithm if policy_stable is true(policy converged).
        converge： True if policy is stable in the last iteration, False otherwise.
        V is the value function for the last policy.
        converge_iter is a list of convergence indicators of all iterations starting from the first iteration.
        time_iter is a list of run time(ms) of all iterations starting from the first iteration.
    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
                #print (state, a, next_state, prob, reward, V[next_state])
        return A
        
    # Capture statistics for analysis
    converge_iter = []
    time_iter = []
    start = time()
    
    V = np.zeros(env.nS)
    for i in range(max_iter):
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value           
        # Update statistics
        converge = (delta < theta)
        #converge_iter.append(converge)
        converge_iter.append(delta)
        current = time()
        time_spt = (current - start) * 1000.0
        time_iter.append(time_spt)
        start = current            
        # Check if we can stop 
        if early_stop and converge:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, converge, V, converge_iter, time_iter