import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt
#import growspace.envs.growspaceenv

env = gym.make("GrowSpaceEnv-Images-v0")




def random_solver():

    state_space = np.zeros((84, 84))
    print(state_space)
    env.seed()
    env.reset()
    # it +=1
    count = 0
    rewards = []
    timers_step = []
    # dict_of_tips = SortedDict()
    actions = [0, 1, 2]
    b_tips = set()
    for s in range(30):  # same as ppo

        if s == 0:  # first step
            s_t = env.step(env.action_space.sample())  ### s_t --> tips, target, light
            tips = s_t[3]['tips']  # add seperately as they are first set
            b_tips.add(tips)  # put in set for unique values

        else:
            tips = s_t[3]['tips']
        n_tips_to_add = len(tips) - len(b_tips)
        b_tips.add(tips[:-n_tips_to_add])  # add last tips of list
        x_range = [i[0] for i in tips[:-n_tips_to_add]]  # find range of x for linking reward
        y_range = [i[1] for i in tips[:-n_tips_to_add]]
        dist = distance.cdist(coords, s_t[3]['target'], 'euclidean')
        pick
        euclidean
        distance
        look
        at
        branch
        positionement
        within
        light
        range
        decide if light is to
        move
        reward = s_t[1]
        print(x_range)

        # add reward to statespace matrix
        # add

        s_t = env.step(action_t)

        action_t
        picke
        print(tips)

#branches_trimmed = sample(self.branches, MAX_BRANCHING)