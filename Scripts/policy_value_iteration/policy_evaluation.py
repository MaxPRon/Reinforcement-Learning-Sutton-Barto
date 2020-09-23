import gym
import numpy as np
#import tools
import timeit



env = gym.make('FrozenLake-v0')
env.reset()


# Map of the environment
# "SFFF"    0  1  2  3
# "FHFH"    4  5  6  7
# "FFFH"    8  9  10 11
# "HFFG"    12 13 14 15
#
# S = Start (safe)
# F = Frozen (safe)
# H = Hole (unsafe)
# G = Goal (safe)
# 
# Possible Action
# 0 = LEFT
# 1 = DOWN
# 2 = RIGHT
# 3 = UP


# Define parameters for Policy Evaluation

num_of_episodes = 20 
gamma = 0.99 # learning rate 
theta = 0.0001 # stop condition, minimal change of values
size_of_env = env.observation_space.n # size of the state space 


# Define policy
pi = np.ones((env.observation_space.n,env.action_space.n))*0.25 # uniform policy
results_value = np.zeros(size_of_env) # value function [number of episodes][state_space ]
delta_value = np.zeros(size_of_env)
def bellman_update(env, pi, s, V): # environment , policy, evaluation state, Value function
    V[s] = 0
    for a in range(env.action_space.n):
        Ps_ = env.env.P[s][a] # state transition probabiltiy to s' Dim:[num of possible states][probability, s', reward, done(boolean)]
        number_of_possible_states, _ = np.shape(Ps_)
        for n in range(number_of_possible_states):
            # compute the new value funtion
            p = Ps_[n][0]
            s_ = Ps_[n][1]
            r = Ps_[n][2]
            V[s] += (pi[s][a])*(p*(r+gamma*V[s_]))
        
    return V[s]
    
#for episode_i in range(num_of_episodes):
#    env.reset()
#    n = 0
start = timeit.default_timer()

while True:
    #delta = 0
    #delta_max = 0
    for s in range(0,env.observation_space.n):
        v_old = results_value[s]
        v_new = bellman_update(env,pi,s, results_value)
        delta_value[s] = abs(v_old-v_new) #version with 
        #delta = abs(v_old-v_new) 
        #if delta > delta_max:
        #    delta_max = delta       
    if np.max(delta_value) < theta:
    #if delta_max < theta:
        break

end = timeit.default_timer()

print("Runtime:", end - start)  

print(results_value)
                

