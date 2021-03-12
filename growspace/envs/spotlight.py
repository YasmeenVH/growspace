import math
import random
import sys
from enum import IntEnum
from random import sample

import cv2
import geometer
import gym
import numpy as np
import growspace.plants.tree
from numpy.linalg import norm
from scipy.spatial import distance

np.set_printoptions(threshold=sys.maxsize)
# customizable variables by user

BRANCH_THICCNESS = .015
MAX_BRANCHING = 10
DEFAULT_RES = 71
LIGHT_WIDTH = .25
LIGHT_DIF = 250
LIGHT_DISPLACEMENT = .1
LIGHT_W_INCREMENT = .1
MIN_LIGHT_WIDTH = .1
MAX_LIGHT_WIDTH = .5
PATH = '../../scripts/png/mnist_data/mnist_1.png'


def to_int(v):
    return int(round(v))


ir = to_int  # shortcut for function calld

FIRST_BRANCH_HEIGHT = ir(.24 * DEFAULT_RES)
BRANCH_LENGTH = (1 / 9) * DEFAULT_RES


def unpack(w):
    return map(list, zip(*enumerate(w)))


class Features(IntEnum):
    light = 0
    scatter = 1


class GrowSpaceEnvSpotlightMnist(gym.Env):
    def __init__(self, width=DEFAULT_RES, height=DEFAULT_RES, path=PATH):
        self.width = width
        self.height = height
        self.seed()
        self.action_space = gym.spaces.Discrete(5)  # L, R, keep of light paddle, or increase, decrease
        self.feature_maps = np.zeros((len(Features), self.height, self.width), dtype=np.uint8)

        self.observation_space = gym.spaces.Box(0, 255, shape=(28, 28, 3), dtype=np.uint8)
        self.mnist_shape = cv2.imread(path)

        self.focus_point = np.array([0.5, 0.5])
        self.focus_radius = 0.1
        self.draw_spotlight()

        self.branches = None
        self.target = None
        self.steps = None
        self.new_branches = None
        self.tips_per_step = None
        self.tips = None

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def light_scatter(self):
        filter_ = np.logical_and(self.feature_maps[Features.light], self.feature_maps[Features.scatter])
        return np.argwhere(filter_)

    def tree_grow(self, activated_photons, mindist, maxdist):
        branches_trimmed = self.branches
        for i in range(len(activated_photons) - 1, 0, -1):  # number of possible scatters, check if they allow for branching with min_dist
            closest_branch = 0
            dist = 1 * self.width

            if len(self.branches) > MAX_BRANCHING:
                branches_trimmed = sample(self.branches, MAX_BRANCHING)
            else:
                branches_trimmed = self.branches

            for branch in branches_trimmed:
                if self.feature_maps[Features.light][branch.tip_point[::-1]]:
                    photon_ptx = activated_photons[i]
                    tip_to_scatter = norm(photon_ptx - branch.tip_point)
                    if tip_to_scatter < dist:
                        print(closest_branch, tip_to_scatter)
                        dist = tip_to_scatter
                        closest_branch = branch

            # removes scatter points if reached

            if dist < mindist:
                activated_photons = np.delete(activated_photons, i)

            # when distance is greater than max distance, branching occurs to find other points.
            elif dist < maxdist:
                closest_branch.grow_count += 1
                branch_length = BRANCH_LENGTH / dist
                photon = activated_photons[i]
                g = (photon - closest_branch.p2) * branch_length
                closest_branch.grow += g

        for branch in branches_trimmed:
            if branch.grow_count > 0:
                x2 = branch.x2 + branch.grow_x / branch.grow_count
                y2 = branch.y2 + branch.grow_y / branch.grow_count
                # TODO: clip if needded

                newBranch = growspace.plants.tree.PixelBranch(branch.x2, ir(x2), branch.y2, ir(y2), self.width, self.height)
                self.branches.append(newBranch)
                branch.child.append(newBranch)
                branch.grow_count = 0
                branch.grow_direction.fill(0)

        # increase thickness of first elements added to tree as they grow

        self.branches[0].update_width()

        branch_coords = []

        # sending coordinates out
        for branch in self.branches:
            branch_coords.append([branch.tip_point])

        self.tips = branch_coords
        return branch_coords

    def distance_target(self, coords):
        dist = distance.cdist(coords, [self.target], 'euclidean')
        min_dist = min(dist)
        return min_dist

    def get_observation(self, debug_show_scatter=False):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        yellow = (0, 128, 128)  # RGB color (dark yellow)
        blue = (255, 0, 0)
        img[self.feature_maps[Features.light].nonzero()] = yellow
        cv2.circle(img, tuple(self.to_image(self.focus_point)), int(self.focus_radius * self.height), (0, 255, 0), thickness=2)

        if debug_show_scatter:
            pts = self.light_scatter()
            for p in pts:
                x, y = p
                cv2.circle(img, center=(y, x), radius=2, color=(255, 0, 0), thickness=-1)

        # Draw plant as series of lines (1 branch = 1 line)
        for branch in self.branches:
            thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
            cv2.line(img, pt1=branch.p, pt2=branch.tip_point, color=(0, 255, 0), thickness=thiccness)

        z = np.where(self.mnist_shape < 255, img, 255)
        # flip image, because plant grows from the bottom, not the top
        img = cv2.flip(z, 0)
        return img

    def reset(self):
        random_start = random.randint(0, self.width - 1)
        self.branches = [
            growspace.plants.tree.PixelBranch(x=random_start, x2=random_start, y=0, y2=FIRST_BRANCH_HEIGHT, img_width=self.width, img_height=self.height)
        ]
        self.target = np.array([np.random.randint(0, self.width), ir(.8 * self.height)])

        x_scatter = np.random.randint(0, self.width, LIGHT_DIF)
        y_scatter = np.random.randint(0, self.height, LIGHT_DIF)
        self.feature_maps[Features.scatter].fill(False)
        self.feature_maps[Features.scatter][y_scatter, x_scatter] = True

        self.steps = 0
        self.new_branches = 0
        self.tips_per_step = 0
        self.tips = [self.branches[0].tip_point, ]

        self.draw_spotlight()
        return self.get_observation()

    def step(self, action):
        if action == 0:
            self.move_sun(0.1)

        if action == 1:
            self.move_sun(-0.1)

        if action == 2:
            self.focus_radius = min(0.2, self.focus_radius + 0.05)

        if action == 3:
            self.focus_radius = max(0.05, self.focus_radius - 0.05)

        if action == 5:
            self.focus_point.array[0] -= 0.1

        if action == 6:
            self.focus_point.array[0] += 0.1

        if action == 7:
            self.focus_point.array[1] += 0.1

        if action == 8:
            self.focus_point.array[1] -= 0.1

        if action == 4:
            pass
        self.draw_spotlight()

        pts = self.light_scatter()
        tips = self.tree_grow(pts, .01 * self.width, .15 * self.width)

        if self.distance_target(tips) <= 0.1:
            reward = 1 / 0.1 / 10
        else:
            reward = 1 / self.distance_target(tips) / 10

        observation = self.get_observation()  # image
        done = False  # because we don't have a terminal condition
        misc = {"tips": tips, "target": self.target, "light": None}

        if self.steps == 0:
            self.new_branches = len(tips)
            misc['new_branches'] = self.new_branches

        else:
            new_branches = len(tips) - self.new_branches
            misc['new_branches'] = new_branches
            self.new_branches = len(tips)  # reset for future step

        misc['img'] = observation
        self.steps += 1
        return observation, reward, done, misc

    def move_sun(self, angle_change):
        self.sun_angle = (self.sun_angle + angle_change) % (2 * math.pi)

    @property
    def sun_position(self):
        return geometer.Point(
            np.around(self.sun_distance * np.cos(self.sun_angle), 5),
            np.around(self.sun_distance * np.sin(self.sun_angle), 5),
        )

    def draw_spotlight(self):
        self.feature_maps[Features.light].fill(False)
        cv2.circle(self.feature_maps[Features.light], tuple(self.to_image(self.focus_point)), int(self.focus_radius * self.height), True, thickness=-1)

    def to_image(self, p):
        if hasattr(p, "normalized_array"):
            return np.around((self.height, self.width) * p.normalized_array[:-1]).astype(np.int32)
        else:
            y, x = p
            return np.around((self.height * y, self.width * x)).astype(np.int32)

    # def step(self, action):
    #     return observation, reward, done, misc


if __name__ == '__main__':
    gse = gym.make('GrowSpaceSpotlight-Mnist-v0')


    def key2action(key):
        if key == ord('a'):
            return 0  # move left
        elif key == ord('d'):
            return 1  # move right
        elif key == ord('w'):
            return 2
        elif key == ord('s'):
            return 3
        elif key == ord('x'):
            return 4
        elif key == ord('h'):
            return 5
        elif key == ord('l'):
            return 6
        elif key == ord('j'):
            return 7
        elif key == ord('k'):
            return 8
        else:
            return None


    rewards = []
    print('what is this')
    while True:
        gse.reset()
        img = gse.get_observation(debug_show_scatter=False)
        # image = img.astype(np.uint8)
        # backtorgb = image * 255
        # print(backtorgb)
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
