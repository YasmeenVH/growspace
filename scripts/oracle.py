import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
env = gym.make("GrowSpaceEnv-Images-v0")


class Oracle(object):
    def __init__(self):
        pass
    def oracle(self):
        state_space = np.zeros((84,84))
        #print(state_space)
        env.seed()
        start = env.reset()
        #it +=1
        count = 0
        rewards = []
        timers_step = []
        #dict_of_tips = SortedDict()
        actions = [0, 1, 2]
        #b_tips = set()
        target_x = env.target[0]
        #rewards = set()
## remember the dictionary for misc is : {"tips": tips, "target": self.target, "light": self.x1_light, "new_branches: self.new_banches}
        for s in range(30):        # same as ppo
            print("this is branch:",env.branches[:])
            #if s == 0:             # first step is different as tips are logged after each step
            if target_x < env.x1_light:
                s_t = env.step(0)
            elif target_x > env.x2_light:
                s_t = env.step(1)
            else:
                s_t = env.step(2)
            print("is this reward:", s_t[1])
            print(env.x1_light)
            rewards.append(s_t[1])
        return rewards
        #print(rewards)

            # else:
            #
            #     tips = s_t[3]['tips']
            # n_tips_to_add = len(tips) - len(b_tips)
            # print([b_tips])
            # #print([:-n_tips_to_add])
            # #b_tips.add(tips[:-n_tips_to_add])  # add last tips of list
            # x_range = [i[0] for i in tips[:-n_tips_to_add]]  # find range of x for linking reward
            # y_range = [i[1] for i in tips[:-n_tips_to_add]]
            # dist = distance.cdist(b_tips[-n_tips_to_add], s_t[3]['target'],'euclidean')
            # rewards.append(dist)
            # index = min(dist)
            # for i in range(len(dist)):
            #     if dist[i] == index:
            #         best_tip = n_tips_to_add[i]
            #
            #
            #





            #pick euclidean distance
            #look at branch positionement within light range
            #decide if light is to move
            #reward = s_t[1]
            #print(x_range)

            #add reward to statespace matrix
            #add

            # s_t = env.step(action_t)








if __name__ == '__main__':
    Oracle = Oracle()
    run_oracle = Oracle.oracle()