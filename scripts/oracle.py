import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os, sys
import imageio
import cv2
#env = gym.make("GrowSpaceEnv-Images-v0")
from scripts.random_solver import random_solver
from array import *
from itertools import chain
from scripts.save_img_movie import save_file_movie_oracle, filter, natural_keys
#from growspace.envs.growspaceenv import GrowSpaceEnv

class Oracle(object):
    def __init__(self):
        #self.env = GrowSpaceEnv()
        pass
    def oracle(self):
        env.seed()
        env.reset()
        rewards = []
        target_x = env.target[0]
        target = env.target

## remember the dictionary for misc is : {"tips": tips, "target": self.target, "light": self.x1_light, "new_branches: self.new_banches}
        for s in range(50):        # same as ppo
            #print("this is branch:",env.branches[:])
            #print(s)
            if s == 0:             # first step is different as tips are logged after each step
                if target_x < env.x1_light:
                    s_t = env.step(0)  # left
                elif target_x > env.x2_light:
                    s_t = env.step(1)  # right
                else:
                    s_t = env.step(2)  # stay
            else:
                n_new_tips = s_t[3]['new_branches']
                #print(n_new_tips)

                if n_new_tips == 0:  #if no new branches ,
                    # get light back to plant
                    # look for where tips are in x_range
                    x_range = [i[0] for i in s_t[3]['tips']]
                    dist_x = [target_x - i for i in x_range]
                    min_dist = min(dist_x)
                    if min_dist < 0:
                        s_t = env.step(0)
                    else:
                        s_t = env.step(1)
                else:
                    #print("this is n_new_tips",n_new_tips)
                    new_tip_coords = s_t[3]['tips'][-n_new_tips:]
                    #print(new_tip_coords)
                    #print('target:', target)
                    dist = distance.cdist(new_tip_coords, [target], 'euclidean')
                    min_dist = min(dist)
                    index_x = [i for i, j in enumerate(dist) if j == min_dist]
                    #print("index;", index_x)
                    #print(new_tip_coords[index_x[0]])
                    light_clue = new_tip_coords[index_x[0]][0] - env.x1_light   # only in x
                    target_clue = target_x - new_tip_coords[index_x[0]][0]
                    if target_clue > 0 and light_clue < 0.1:
                        s_t = env.step(2)
                    elif target_clue > 0 and light_clue > 0.1:
                        s_t = env.step(1)
                    else:
                        s_t = env.step(0)

            img = env.get_observation(debug_show_scatter=False)

            path = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/'
            cv2.imwrite(os.path.join(path, 'step_oracle_good' + str(s) + '.png'), img)
            print(s_t[1])
            rewards.append(s_t[1])
            #rewards = rewards.tolist()
        flatten_list = list(chain.from_iterable(rewards))

        return flatten_list


if __name__ == '__main__':
    env = gym.make("GrowSpaceEnv-Images-v0")
    Oracle = Oracle()
    run_oracle = Oracle.oracle()
    png_dir = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/'
    list_filter = filter(png_dir)
    list_filter.sort(key=natural_keys)
    print(list_filter)
    save_file_movie_oracle(list_filter)

    # #print(sum(run_oracle))
    # timers_reset = []
    # timings = []
    # episodes = 100
    # random_rewards = []
    # oracle_rewards = []
    # for _ in range(episodes):  # here we're timing the reset function
    #     run_oracle = Oracle.oracle()
    #     #flat_list = [[item] for item in run_oracle]
    #     print("runoracle:", run_oracle)
    #     oracle_rewards.append(run_oracle)
    #
    # for _ in range(episodes):
    #     random_s = random_solver(50)
    #     random_rewards.append(random_s)
    #
    # av_oracle = np.mean(oracle_rewards, axis=0)
    # print("these are av rewards:",av_oracle)
    # err_oracle = np.std(oracle_rewards, axis=0)
    #
    # av_random = np.mean(random_rewards, axis=0)
    # err_random = np.std(random_rewards, axis=0)
    # x = np.arange(0,50)
    #
    # fig, ax = plt.subplots(1)
    # ax.plot(x, av_oracle, color = 'green')
    #
    # ax.plot(x, av_random, color = 'magenta')
    # ax.fill_between(x, av_oracle+err_oracle, av_oracle-err_oracle, alpha=0.5, color='green')
    # ax.fill_between(x, av_random + err_random, av_random - err_random, alpha=0.5, color='magenta')
    # ax.set_title(r'over 100 episodes and 50 steps for oracle and random')
    # ax.set_ylabel("rewards")
    # ax.set_xlabel("steps")
    # ax.grid()
    # plt.show()