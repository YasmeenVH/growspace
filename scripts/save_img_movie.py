import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import imageio
import cv2

env = gym.make("GrowSpaceEnv-Images-v0")

def run_episode():
    env.reset()
    for step in range(25):        # defined in __init__.py    # here we're timing the step function
        env.step(env.action_space.sample())
        img = env.get_observation(debug_show_scatter=True)
        path = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/'
        cv2.imwrite(os.path.join(path,'step'+str(step)+'.png'),img)


def save_file_movie():
    png_dir = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/'
    images = []
    step = 0
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.png') and file_name =='step{}.png'.format(step):

            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
            step+=1
    imageio.mimsave('../scripts/movie/movie.gif', images,fps=1)

if __name__ == '__main__':
  #run_episode()
  save_file_movie()

