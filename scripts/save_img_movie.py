import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import imageio
import cv2
import re
#from scripts.oracle import Oracle
env = gym.make("GrowSpaceEnv-Images-v0")

def run_episode():
    env.reset()
    for step in range(25):        # defined in __init__.py    # here we're timing the step function
        env.step(env.action_space.sample())
        img = env.get_observation(debug_show_scatter=True)
        path = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/'
        cv2.imwrite(os.path.join(path,'step_oracle'+str(step)+'.png'),img)


def filter(png_dir):
    step = 0
    movie_files = []
    print(os.listdir(png_dir))
    for i in range(0, len(os.listdir(png_dir))):
        if os.listdir(png_dir)[i].startswith("step_oracle_good"):
            movie_files.append(os.listdir(png_dir)[i])
            #print("[ng_di i",os.listdir(png_dir)[i])
    #files = sorted(movie_files)
    #print("these are files",files)
    return movie_files

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def save_file_movie():
    png_dir = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/'
    images = []
    step = 0
    #print(os.listdir(png_dir))
    for file_name in os.listdir(png_dir):
        if file_name.startswith('step_oracle') and file_name =='step_oracle{}.png'.format(step):
            print(file_name)
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
            step+=1

    imageio.mimsave('../scripts/movie/movie_oracle_other.gif', images,fps=1)

def save_file_movie_oracle(movie_list):
    png_dir = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/'
    images = []
    step = 0
    #print(os.listdir(png_dir))
    for file_name in movie_list:

        file_path = os.path.join(png_dir, file_name)
        print(file_path)
        images.append(imageio.imread(file_path))
        #print(file_name)


    imageio.mimsave('../scripts/movie/movie_oracle_other.gif', images, fps=0.7)
if __name__ == '__main__':
  #run_episode()


  save_file_movie()

