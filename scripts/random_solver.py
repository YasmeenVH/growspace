import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt
#import growspace.envs.growspaceenv
from itertools import chain


env = gym.make("GrowSpaceEnv-Images-v0")




def random_solver(num_steps):

    state_space = np.zeros((84, 84))
    print(state_space)
    env.seed()
    env.reset()

    count = 0
    rewards = []

    for s in range(num_steps):  # same as ppo

        s_t = env.step(env.action_space.sample())  ### s_t --> tips, target, light
        rewards.append(s_t[1])
    flatten_list = list(chain.from_iterable(rewards))

    return flatten_list

if __name__ == '__main__':
    random=random_solver(30)
    print(random)