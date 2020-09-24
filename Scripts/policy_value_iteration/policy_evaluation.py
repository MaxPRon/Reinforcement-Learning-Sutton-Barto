import gym
import numpy as np
import timeit
import matplotlib as mp
import matplotlib.pyplot as plt


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
runtime = np.zeros(num_of_episodes)
gamma = 0.99 # learning rate 
theta = 0.0001 # stop condition, minimal change of values
size_of_env = env.observation_space.n # size of the state space 


# Define policy
pi = np.ones((env.observation_space.n,env.action_space.n))*0.25 # uniform policy

# Value function array
results_value = np.zeros((num_of_episodes,size_of_env)) # value function [number of episodes][state_space ]


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
    
for episode_i in range(0,num_of_episodes):
    env.reset()
    start = timeit.default_timer()

    while True:
        delta = 0
        delta_max = 0
        #delta_value = np.zeros(size_of_env) # computational slower as compared to the 2 variable version

        for s in range(0,env.observation_space.n):
            
            v_old = results_value[episode_i][s]
            v_new = bellman_update(env,pi,s, results_value[episode_i])
            #delta_value[s] = abs(v_old-v_new) #version with longer runtime 
            delta = abs(v_old-v_new) 
            if delta > delta_max:
                delta_max = delta
        #if np.max(delta_value) < theta: 
        if delta_max < theta:
            break

    end = timeit.default_timer()
    runtime[episode_i] = end-start
    

result_square = np.round(np.reshape(results_value[0],(4,4)),5)
_, ax = plt.subplots()
plt.imshow(result_square)
plt.title('Value Function of the given policy')
plt.axis('off')
plt.set_cmap('plasma')
for i in range(4):
    for j in range(4):
        text = ax.text(j, i, result_square[i, j],
                       ha="center", va="center", color="w")
plt.colorbar()
plt.show(block =False)
plt.pause(10)
plt.close()


print("The average runtime for computation is: {}s ".format(np.average(runtime)))
print("The shortest computation time is {}s".format(np.min(runtime)))
print("The longest computation time is {}s".format(np.max(runtime)))
print("The standard deviation of the computation time is {}s".format(np.std(runtime)))


