import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os, sys
import imageio
import cv2


from scripts.save_img_movie import save_file_movie_oracle, filter, natural_keys
#from growspace.envs.growspaceenv import GrowSpaceEnv

class OracleMnist(object):
    def __init__(self):
        pass

    def oracle(self):
        env.seed()
        env.reset()
        rewards = []
        mnist_map = env.


## remember the dictionary for misc is : {"tips": tips, "target": self.target, "light": self.x1_light, "new_branches: self.new_banches}
        for s in range(50):
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
            cv2.imwrite(os.path.join(path, 'step_oracle_mnist1_' + str(s) + '.png'), img)
            print(s_t[1])
            rewards.append(np.float(s_t[1]))
            #rewards = rewards.tolist()
        print(rewards)
        #rewards_good = [[i] for i in rewards]

#       #flatten_list = list(chain.from_iterable(rewards))
        flatten_list = np.sum(rewards)
        print('total r:',flatten_list)

        return flatten_list


if __name__ == '__main__':
    env = gym.make("GrowSpaceSpotlight-Mnist1-v0")
    Oracle = OracleMnist()
    run_oracle = Oracle.oracle()
    png_dir = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/oraclemnist'
    list_filter = filter(png_dir)
    list_filter.sort(key=natural_keys)
    print(list_filter)
    save_file_movie_oracle(list_filter)
