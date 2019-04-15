"""
Policy iteration algorithm to find the optimal policies for reinforcement learning problem
adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
"""

import numpy as np
from time import time 

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


def policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=1.0, max_iter=100, early_stop=True):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
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
        return A
    
    # Capture statistics for analysis
    converge_iter = []
    time_iter = []
    start = time()
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    for i in range(max_iter):
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        
        # Will be set to false if we make any changes to the policy
        converge = True
        
        # For each state...
        delta = 0
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)
            best_a_value = np.max(action_values)
            
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_a_value - V[s]))
            
            # Greedily update the policy
            if chosen_a != best_a:
                converge = False
            policy[s] = np.eye(env.nA)[best_a]
            
        # Update statistics
        #converge_iter.append(converge)
        converge_iter.append(delta)
        current = time()
        time_spt = (current - start) * 1000.0
        time_iter.append(time_spt)
        start = current
        
        # If the policy is stable we've found an optimal policy. Return it
        if early_stop and converge:
            return policy, converge, V, converge_iter, time_iter
     
    return policy, converge, V, converge_iter, time_iter
            