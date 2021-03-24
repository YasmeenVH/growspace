mport gym
import numpy as np
from scipy.spatial import distance
import os
import cv2
#env = gym.make("GrowSpaceEnv-Images-v0")
from itertools import chain
from scripts.save_img_movie import save_file_movie_oracle, filter, natural_keys
#from growspace.envs.growspaceenv import GrowSpaceEnv

class Oracle_Fairness(object):
    def __init__(self):
        #self.env = GrowSpaceEnv()
        pass
    def check(self, plant1, plant2, target_x):
        info = []
        # condition 1 is plant1 under light
        if env.x1_light <= plant1[0].x <= env.x2_light:
           info.append(1)
        else:
            info.append(0)
        # condition 2 is plant1 under light
        if env.x2_light <= plant2[0].x <= env.x2_light:
            info.append(1)
        else:
            info.append(0)
        # condition 3 is plant 2 greater than plant 1
        if plant1 < plant2:
            info.append(1)
        else:
            info.append(0)

       #when plant1 is not under light
        if info[0] == 0:
            # when plant2 is not under light
            if info[1] == 0:
                # when plant1 is smaller than light 1
                if plant1[0].x < env.x1_light:
                    # when plant1 bigger than plant2
                    if info[3] == 0:
                        return "a"
                    # when plant2 bigger than plant1 light 1 and light is right of plants
                    elif plant2 < env.x1_light:
                        return "a"   # light is left of plants
                    else: # plant2 is greater and plant2 is bigger than light positionment
                        return "b" # light is in between plants
                elif plant1[0].x < env.x1_light:

                    return "b" # light is in between plants


    # confirm plant2 is under light
    def oracle_fair(self):
        '''
        remember the dictionary for misc is : {"tips": tips, "target": self.target, "light": self.x1_light, "new_branches: self.new_banches}
        tips = branch_coords, branch_coords2   where branch_coords = [x, y]
        actions: 0 = left, 1 = right, 2 = increase, 3 = decrease, 4 stay
        :return:
        '''
        env.seed()
        env.reset()
        rewards = []
        target_x = env.target[0]
        target = env.target
        plant1 = env.branches
        plant2 = env.branches2
        d = np.abs(plant1[0].x - plant2[0].x)
        light_w = env.light_width
        if plant1[0].x < plant2[0].x

        for s in range(50):        # same as ppo
            #print("this is branch:",env.branches[:])
            #print(s)
            if s == 0:             # first step is different as tips are logged after each step
                ## check where light is positioned lef
                if env.x1_light <= plant1[0].x <= env.x2_light:
                    # confirms plant1 is under light

                if env.x2_light <= plant2[0].x <= env.x2_light:
                    # confirm plant2 is under light
                if env.x1_light <= plant1[0].x and env.x1_light <= plant2[0].x:
                    if env.x2_light => plant1[0].x and env.x2_light => plant2[0].x:

                    else:
                else:

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
            cv2.imwrite(os.path.join(path, 'step_oracle_fair' + str(s) + '.png'), img)
            print(s_t[1])
            rewards.append(s_t[1])
            #rewards = rewards.tolist()
        flatten_list = list(chain.from_iterable(rewards))

        return flatten_list


if __name__ == '__main__':
    env = gym.make("GrowSpaceEnv-Fairness-v0")
    Oracle = Oracle_Fairness()
    run_oracle = Oracle.oracle_fair()
    png_dir = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/'
    list_filter = filter(png_dir)
    list_filter.sort(key=natural_keys)
    print(list_filter)
    save_file_movie_oracle(list_filter)

