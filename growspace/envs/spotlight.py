import os
import random
import sys
from enum import IntEnum
from random import sample

import cv2
import gym
import numpy as np
import tqdm
from numpy.linalg import norm
from scipy.spatial import distance
import random

import growspace.plants.tree

np.set_printoptions(threshold=sys.maxsize)
# customizable variables by user

BRANCH_THICCNESS = .015 # before was 0.036 in 28 x 28
MAX_BRANCHING = 10
DEFAULT_RES = 84
LIGHT_WIDTH = 0.25
LIGHT_DIF = 200
LIGHT_DISPLACEMENT = 0.1
LIGHT_W_INCREMENT = 0.1
MIN_LIGHT_WIDTH = 0.1
MAX_LIGHT_WIDTH = 0.5

BRANCH_LENGTH = (1 / 9) * DEFAULT_RES
PATH = os.path.dirname(__file__) + "/../../scripts/png/mnist_data/"

"""digit is the mnist number we want to pass
enter as a string. for mix combos enter as : 1_7_mix
refer to directory names
"""

# import config
# for k in list(locals()):
#     if f"^" + k in config.tensorboard.run.config:
#         locals()[k] = config.tensorboard.run.config[f"^" + k]

def to_int(v):
    return int(round(v))


ir = to_int  # shortcut for function calld

FIRST_BRANCH_HEIGHT = ir(0.1 * DEFAULT_RES)


def unpack(w):
    return map(list, zip(*enumerate(w)))

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            dsize = (84,84) # try
            img = cv2.resize(img, dsize) #try
            images.append(img)
    return images

class Features(IntEnum):
    light = 0
    scatter = 1


class GrowSpaceEnvSpotlightMnist(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}  # Required, otherwise gym.Monitor disables itself.

    def __init__(self, width=DEFAULT_RES, height=DEFAULT_RES, path=PATH, digit='3'):
        self.width = width
        self.height = height
        self.seed()
        self.action_space = gym.spaces.Discrete(5)  # L, R, keep of light paddle, or increase, decrease
        self.feature_maps = np.zeros((len(Features), self.height, self.width), dtype=np.uint8)

        self.observation_space = gym.spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.digit = digit
        self.path = path
        #assert os.path.isfile(path), "path to mnist image is not valid"
        if self.digit =='curriculum':
            self.shape = None
        else:
            self.shape = path + digit +'/'

        #self.mnist_shape = cv2.imread(path)

        self.focus_point = None
        self.focus_radius = None

        self.branches = None
        self.target = None
        self.steps = None
        self.new_branches = None
        self.tips_per_step = None
        self.tips = None
        self.episode = -1

    def render(self, mode='human', debug_show_scatter=False):
        """
        @mode: ['mode', 'rgb_array']

        """
        img = self.get_observation(debug_show_scatter)
        #dsize = (84,84)
        #img = cv2.resize(img, dsize)
        # if self.obs_type == 'Binary':
        #     image = img.astype(np.uint8)
        #     img = image * 255

        if mode == "human":
            cv2.imshow('plant', img)  # create opencv window to show plant
            cv2.waitKey(1)  # this is necessary or the window closes immediately
        else:
            return img


    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def light_scatter(self):
        filter_ = np.logical_and(self.feature_maps[Features.light], self.feature_maps[Features.scatter])
        return np.argwhere(filter_)

    def tree_grow(self, activated_photons, mindist, maxdist):
        branches_trimmed = self.branches
        # number of possible scatters, check if they allow for branching with min_dist
        for i in range(len(activated_photons) - 1, 0, -1):
            closest_branch = 0
            dist = 1 * self.width

            if len(self.branches) > MAX_BRANCHING:
                branches_trimmed = sample(self.branches, MAX_BRANCHING)
            else:
                branches_trimmed = self.branches

            for branch in branches_trimmed:
                # if self.feature_maps[Features.light][branch.tip_point[::-1]]: # this was alsready done in light scatter func
                photon_ptx = np.flip(activated_photons[i])  # flip was necessary bc coordinate systems are inverted -
                tip_to_scatter = norm(photon_ptx - np.array(branch.tip_point))  # one is np.array, one is tuple
                if tip_to_scatter < dist:
                    dist = tip_to_scatter
                    closest_branch = branch

            # removes scatter points if reached

            if dist < mindist:
                activated_photons = np.delete(activated_photons, i)

            # when distance is greater than max distance, branching occurs to find other points.
            elif dist < maxdist:
                closest_branch.grow_count += 1
                branch_length = BRANCH_LENGTH / dist
                photon = np.flip(activated_photons[i])
                g = (photon - closest_branch.tip_point) * branch_length
                closest_branch.grow_direction += np.round(g).astype(np.int)

        for branch in branches_trimmed:
            if branch.grow_count > 0:
                (x2, y2) = branch.tip_point + branch.grow_direction / branch.grow_count
                x2 = np.clip(x2, 0, self.width - 1)
                y2 = np.clip(y2, 0, self.height - 1)

                newBranch = growspace.plants.tree.PixelBranch(
                    branch.x2, ir(x2), branch.y2, ir(y2), self.width, self.height
                )
                self.branches.append(newBranch)
                branch.child.append(newBranch)
                branch.grow_count = 0
                branch.grow_direction.fill(0)

        # increase thickness of first elements added to tree as they grow

        self.branches[0].update_width()

        branch_coords = []

        # sending coordinates out
        for branch in self.branches:
            branch_coords.append(branch.tip_point)

        self.tips = branch_coords
        return branch_coords

    # def distance_target(self, coords):
    #     dist = distance.cdist(coords, [self.target], "euclidean")
    #     min_dist = min(dist)
    #     return min_dist

    def get_observation(self, debug_show_scatter=False):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        yellow = (0, 128, 128)  # RGB color (dark yellow)

        img[self.feature_maps[Features.light].nonzero()] = yellow
        cv2.circle(img, tuple(self.to_image(self.focus_point)), int(self.focus_radius * self.height), (0, 128, 128), thickness=2) # only contour

        if debug_show_scatter:
            pts = self.light_scatter()
            for p in pts:
                x, y = p
                cv2.circle(img, center=(y, x), radius=2, color=(255, 0, 0), thickness=-1)

        # Draw plant as series of lines (1 branch = 1 line)
        for branch in self.branches:
            thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
            cv2.line(img, pt1=branch.p, pt2=branch.tip_point, color=(0, 255, 0), thickness=thiccness)

        img = cv2.flip(img,0)
        z = np.where(self.mnist_shape < 255, img, 150)
        # flip image, because plant grows from the bottom, not the top
        #img = cv2.flip(z, 0)
        return z #img

    def reset(self):
        random_start = random.randint(self.width - (self.width*3/4), self.width - 1 - (self.width*1/4))
        self.branches = [
            growspace.plants.tree.PixelBranch(
                x=random_start,
                x2=random_start,
                y=0,
                y2=FIRST_BRANCH_HEIGHT,
                img_width=self.width,
                img_height=self.height,
            )
        ]
        #self.target = np.array([np.random.randint(0, self.width), ir(0.8 * self.height)])
        #self.mix = load_images(self.shape_1) + load_images(self.shape_7)

        #flat_list = [item for sublist in self.mix for item in sublist]
        self.episode += 1
        if self.digit == 'curriculum':
            self.path = PATH

            if self.episode < 500:
                self.shape = self.path + '36' + '/'
                if self.episode == 1:
                    print('check1')

            if 500 <= self.episode < 1000:
                self.shape = self.path + '362' + '/'
                if self.episode == 500:
                    print('check2')

            if 1000 <= self.episode < 1500:
                self.shape = self.path + '3621' + '/'
                if self.episode == 1000:
                    print('check3')

            if 1500 <= self.episode < 2500:
                self.shape = self.path + '36214' + '/'
                if self.episode == 1500:
                    print('check4')

            if 2500 <= self.episode < 3500:
                self.shape = self.path + '362145' + '/'
                if self.episode == 2500:
                    print('check5')

            if 3500 <= self.episode < 5000:
                self.shape = self.path + '3621457' + '/'
                if self.episode == 3500:
                    print('check6')

            if 5000 <= self.episode < 6500:
                self.shape = self.path + '36214578' + '/'
                if self.episode == 5000:
                    print('check7')

            if 6500 <= self.episode < 10000:
                self.shape = self.path + 'partymix' + '/'
                if self.episode == 6500:
                    print('check8')

        if self.digit == 'curriculum2':
            self.path = PATH

            if self.episode < 500:
                self.shape = self.path + '36' + '/'
                if self.episode == 1:
                    print('check1')

            if 500 <= self.episode < 1000:
                self.shape = self.path + '362' + '/'

                if self.episode == 500:
                    print('check2')

            if 1000 <= self.episode < 1500:
                self.shape = self.path + '3621' + '/'
                if self.episode == 1000:
                    print('check3')

            if 1500 <= self.episode < 2500:
                self.shape = self.path + '36214' + '/'
                if self.episode == 1500:
                    print('check4')

            if 2500 <= self.episode < 3500:
                self.shape = self.path + '362145' + '/'
                if self.episode == 2500:
                    print('check5')

            if 3500 <= self.episode < 5000:
                self.shape = self.path + '3621457' + '/'
                if self.episode == 3500:
                    print('check6')

                if 5000 <= self.episode < 6500:
                    self.shape = self.path + '36214578' + '/'
                    if self.episode == 5000:
                        print('check7')

                if 6500 <= self.episode < 10000:
                    self.shape = self.path + 'partymix' + '/'
                    if self.episode == 6500:
                        print('check8')

        self.shapes = load_images(self.shape)
        self.mnist_shape = random.choice(self.shapes)
        #print(len(mnist_shapes))
        self.focus_point = np.array([random_start / self.width, FIRST_BRANCH_HEIGHT / self.height])
        self.focus_radius = 0.1

        x_scatter = np.random.randint(0, self.width, LIGHT_DIF)
        y_scatter = np.random.randint(0, self.height, LIGHT_DIF)
        self.feature_maps[Features.scatter].fill(False)
        #print('what is this',self.feature_maps)

        self.feature_maps[Features.scatter][y_scatter, x_scatter] = True
        #print('what is this again', self.feature_maps[0])
        self.steps = 0
        self.new_branches = 0
        self.tips_per_step = 0
        self.tips = [
            self.branches[0].tip_point,
        ]

        self.draw_spotlight()
        #print('showme this',self.feature_maps)
        self.mnist_pixels = (self.get_observation()[:, :, 2] / 150)  # binary map of mnist shape

        plant_stem = (self.get_observation()[:, :, 1] / 255)
        plant_stem[plant_stem>0.6] =1              # filter for green
        self.plant_original = plant_stem.astype(int)


        return self.get_observation()

    def step(self, action):
        if action == 0:
            self.focus_radius = min(0.2, self.focus_radius + 0.05)

        if action == 1:
            self.focus_radius = max(0.03, self.focus_radius - 0.05)

        if action == 2:
            self.focus_point[0] = max(0, self.focus_point[0] - 0.1)

        if action == 3:
            self.focus_point[0] = min(1, self.focus_point[0] + 0.1)

        if action == 4:
            self.focus_point[1] = min(1, self.focus_point[1] + 0.1)

        if action == 5:
            self.focus_point[1] = max(0, self.focus_point[1] - 0.1)

        if action == 6:
            pass
        #self.draw_spotlight()

        pts = self.light_scatter()
        tips = self.tree_grow(pts, 0.01 * self.width, 0.15 * self.width)
        self.draw_spotlight()
        observation = self.get_observation()  #image

        plant = (observation[:,:,1]/255) # binary map of plant
        pixel_plant = np.sum(plant)

        plant[plant>0.6] =1              # filter for green
        plant = plant.astype(int)
        true_plant = np.subtract(plant,self.plant_original)


        mnist = (observation[:,:,2]/150)     # binary map of mnist

        mnist[mnist>0.5] =1
        mnist = mnist.astype(int)

        check = np.sum((true_plant, mnist), axis=0)
        #print('this is check', check)

        intersection = np.sum(np.where(check < 2, 0, 1))

        negative_reward = np.subtract(true_plant,mnist)
        punishment = np.where(negative_reward>0, negative_reward,0)
        #print('what is punishment', punishment)
        bad_pixels = np.sum(punishment)*0.0001
        #print('this is bad_pixels', bad_pixels)
        #print("this is nega",negative_reward)

        union = np.sum(np.where(check<2,check,1))

        reward = intersection / union

        #reward = 1/(1+np.exp(-reward))
        reward = np.log10(reward) # this is equivalent to ln, np.log10

        #print('reqward',reward)
        #reward = reward-bad_pixels

        done = False  # because we don't have a terminal condition
        misc = {"tips": tips, "target": self.target, "light": None}

        if self.steps == 0:
            self.new_branches = len(tips)
            misc["new_branches"] = self.new_branches

        else:
            new_branches = len(tips) - self.new_branches
            misc["new_branches"] = new_branches
            self.new_branches = len(tips)  # reset for future step

        misc["img"] = observation
        misc["plant_pixel"] = pixel_plant
        self.steps += 1
        return observation, float(reward), done, misc

    def draw_spotlight(self):
        self.feature_maps[Features.light].fill(False)
        cv2.circle(
            self.feature_maps[Features.light],
            tuple(self.to_image(self.focus_point)),
            int(self.focus_radius * self.height),
            True,
            thickness=-1,

        )

    def to_image(self, p):
        if hasattr(p, "normalized_array"):
            return np.around((self.height, self.width) * p.normalized_array[:-1]).astype(np.int32)
        else:
            y, x = p
            return np.around((self.height * y, self.width * x)).astype(np.int32)


def enjoy():
    #gse = gym.make("GrowSpaceSpotlight-Mnist1-v0")
    gse = GrowSpaceEnvSpotlightMnist()

    def key2action(key):
        if key == ord("+"):
            return 0  # move left
        elif key == ord("-"):
            return 1  # move right
        elif key == ord("a"):
            return 2
        elif key == ord("d"):
            return 3
        elif key == ord("w"):
            return 4
        elif key == ord("s"):
            return 5
        elif key == ord("x"):
            return 6
        else:
            return None

    while True:
        gse.reset()
        img = gse.get_observation(debug_show_scatter=False)
        cv2.imshow("plant", img)
        cv2.waitKey(-1)
        rewards = []
        for _ in range(50):
            action = key2action(cv2.waitKey(-1))
            if action is None:
                quit()

            b, t, c, f = gse.step(action)
            print(f["new_branches"])
            rewards.append(t)
            cv2.imshow("plant", gse.get_observation(debug_show_scatter=False))
        total = sum(rewards)

        print("amount of rewards:", total)  # cv2.waitKey(1)  # this is necessary or the window closes immediately
        # else:
        # dreturn img


def profile():
    #gse = gym.make("GrowSpaceSpotlight-Mnist1-v0")
    gse = GrowSpaceEnvSpotlightMnist()
    gse.reset()

    def do_step():
        a = gse.action_space.sample()
        s, r, d, i = gse.step(a)
        if d:
            gse.reset()

    for _ in tqdm.trange(100000):
        do_step()
    print("hi")


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    #
    # env = GrowSpaceEnvSpotlightMnist()
    # env.reset()
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(env.feature_maps[Features.light])
    # axs[1].imshow(env.feature_maps[Features.scatter])
    # plt.show()
    #
    # env.draw_spotlight()
    # pts = env.light_scatter()
    # print(pts.shape)
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(env.feature_maps[Features.light])
    # img = np.zeros((71, 71), dtype=np.float)
    # img[pts[:, 0], pts[:, 1]] = True
    # axs[1].imshow(img)
    # plt.show()

    enjoy()
