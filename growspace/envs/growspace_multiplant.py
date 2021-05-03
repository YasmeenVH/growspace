from enum import Enum
from enum import IntEnum
from random import sample

import cv2
import gym
import numpy as np
from numpy.linalg import norm
import time
from growspace.plants.tree import PixelBranch
from numpy.linalg import norm
from scipy.spatial import distance
import sys
import itertools
from sklearn import preprocessing
from numba import jit
from functools import partial
np.set_printoptions(threshold=sys.maxsize)

DEFAULT_RES = 84
BRANCH_THICCNESS = .015
BRANCH_LENGTH = (1/10)*DEFAULT_RES
MAX_BRANCHING = 8
LIGHT_WIDTH = .25
LIGHT_DIF = 200
LIGHT_DISPLACEMENT = .1
LIGHT_W_INCREMENT = .1
MIN_LIGHT_WIDTH = .1
MAX_LIGHT_WIDTH = 1
FIRST_BRANCH_HEIGHT = int(.2*DEFAULT_RES)


class Actions(Enum):
    move_left = 0
    move_right = 1
    increase_beam = 2
    decrease_beam = 3
    noop = 4


def to_int(v):
    return int(round(v))

def unpack(w):
    return map(list, zip(*enumerate(w)))


ir = to_int  # shortcut for function call

class Features(IntEnum):
    light = 0
    scatter = 1

class GrowSpaceEnv_Fairness(gym.Env):

    def __init__(self, width=DEFAULT_RES, height=DEFAULT_RES, light_dif=LIGHT_DIF, obs_type = None, level=None, setting = 'easy'):
        self.width = width
        self.height = height
        self.seed()
        self.light_dif = light_dif
        self.action_space = gym.spaces.Discrete(5)  # L, R, keep of light paddle
        self.obs_type = obs_type
        if self.obs_type == None:
            self.observation_space = gym.spaces.Box(
                0, 255, shape=(84, 84, 3), dtype=np.uint8)
        if self.obs_type == 'Binary':
            self.observation_space = gym.spaces.Box(
                0, 1, shape=(84, 84, 5), dtype=np.uint8)
        self.level = level
        self.setting = setting
        self.__initialized = False
        self.feature_maps = np.zeros((len(Features), self.height, self.width), dtype=np.uint8)
        # note: I moved the code for the first branch into the reset function,
        # because when you start an environment for the first time,
        # you're supposed to call "reset()" first before doing anything else

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def light_scatter(self):
        # select scattering with respect to position of the light

        # select the instances where the conditions is true (where the x coordinate is within the light)
        # filter = np.logical_and(self.x_scatter >= self.x1_light,
        #                         self.x_scatter <= self.x2_light)

        # apply filter to both y and x coordinates through the power of Numpy magic :D
        # ys = self.y_scatter[filter]
        # xs = self.x_scatter[filter]
        #
        filter_ = np.logical_and(self.feature_maps[Features.light], self.feature_maps[Features.scatter])
        return np.argwhere(filter_)


    def light_move_R(self):
        if np.around(self.x1_light + self.light_width,2) <= self.width - LIGHT_DISPLACEMENT*self.width:  # limit of coordinates
            self.x1_light += LIGHT_DISPLACEMENT*self.width  # stay put
        else:
            self.x1_light = self.width - self.light_width

    def light_move_L(self):
        if np.around(self.x1_light,2) >= LIGHT_DISPLACEMENT*self.width:  # limit of coordinates
            self.x1_light -= LIGHT_DISPLACEMENT*self.width
        else:
            self.x1_light = 0  # move by .1 leftdd

    def light_decrease(self):
        if np.around(self.light_width,1) <= MIN_LIGHT_WIDTH*self.width:
            self.light_width = self.light_width
        else:
            self.light_width -= LIGHT_W_INCREMENT*self.width

    def light_increase(self):
        if self.light_width >= MAX_LIGHT_WIDTH*self.width:
            #self.light_width = self.light_width
            pass
        elif self.x1_light + self.light_width >= self.width:
            self.light_width = self.width-self.x1_light
        else:
            self.light_width += LIGHT_W_INCREMENT*self.width


    def tree_grow(self, activated_photons, mindist, maxdist): #

        # apply filter to both idx and branches

        branches_trimmed = self.branches
        branches_trimmed2 = self.branches2
        for i in range(len(activated_photons) - 1, 0, -1):  # number of possible scatters, check if they allow for branching with min_dist
            closest_branch = 0
            dist = 1 * self.width

            if len(self.branches) > MAX_BRANCHING:
                branches_trimmed = sample(self.branches, MAX_BRANCHING)
            else:
                branches_trimmed = self.branches

            for branch in branches_trimmed:
                photon_ptx = np.flip(activated_photons[i])  # Flip for inverted coordinates
                tip_to_scatter = norm(photon_ptx - np.array(branch.tip_point))

                if tip_to_scatter < dist:
                    dist = tip_to_scatter
                    closest_branch = branch

            if dist < mindist:
                activated_photons = np.delete(activated_photons, i)

            # when distance is greater than max distance, branching occurs to find other points.
            elif dist < maxdist:

                closest_branch.grow_count += 1
                branch_length = BRANCH_LENGTH / dist
                photon = np.flip(activated_photons[i])
                g = (photon - closest_branch.tip_point) * branch_length
                closest_branch.grow_direction += np.round(g).astype(np.int)

            # branch_idx = [branch_idx for branch_idx, branch in enumerate(branches_trimmed)if self.x1_light <= branch.x2 <= self.x2_light]
            # temp_dist = [norm([x[i] - branches_trimmed[branch].x2, y[i] - branches_trimmed[branch].y2]) for branch in branch_idx]
            #
            # for j in range(0, len(temp_dist)):
            #     if temp_dist[j] < dist:
            #         dist = temp_dist[j]
            #         closest_branch = branch_idx[j]
            #
            #
            # # removes scatter points if reached
            #
            # if dist < mindist:
            #     x = np.delete(x, i)
            #     y = np.delete(y, i)
            #
            # # when distance is greater than max distance, branching occurs to find other points.
            # elif dist < maxdist:
            #
            #     #self.branches[closest_branch].grow_count += 1
            #     branches_trimmed[closest_branch].grow_count += 1
            #     #self.branches[closest_branch].grow_x += (
            #         #x[i] - self.branches[closest_branch].x2) / (dist/BRANCH_LENGTH)
            #     branches_trimmed[closest_branch].grow_x += (
            #         x[i] - branches_trimmed[closest_branch].x2) / (dist/BRANCH_LENGTH)
            #     #self.branches[closest_branch].grow_y += (
            #         #y[i] - self.branches[closest_branch].y2) / (dist/BRANCH_LENGTH)
            #     branches_trimmed[closest_branch].grow_y += (
            #         y[i] - branches_trimmed[closest_branch].y2) / (dist / BRANCH_LENGTH)

        ## FOR SECOND TREE
        for i in range(len(activated_photons) - 1, 0, -1):  # number of possible scatters, check if they allow for branching with min_dist
            closest_branch = 0
            dist = 1 * self.width

            if len(self.branches2) > MAX_BRANCHING:
                branches_trimmed2 = sample(self.branches2, MAX_BRANCHING)
            else:
                branches_trimmed2 = self.branches2

            for branch in branches_trimmed2:
                photon_ptx = np.flip(activated_photons[i])  # Flip for inverted coordinates
                tip_to_scatter = norm(photon_ptx - np.array(branch.tip_point))

                if tip_to_scatter < dist:
                    dist = tip_to_scatter
                    closest_branch2 = branch

            if dist < mindist:
                activated_photons = np.delete(activated_photons, i)

                # when distance is greater than max distance, branching occurs to find other points.
            elif dist < maxdist:

                closest_branch2.grow_count += 1
                branch_length = BRANCH_LENGTH / dist
                photon = np.flip(activated_photons[i])
                g = (photon - closest_branch2.tip_point) * branch_length
                closest_branch2.grow_direction += np.round(g).astype(np.int)
            # branch_idx = [branch_idx for branch_idx, branch in enumerate(branches_trimmed2) if
            #               self.x1_light <= branch.x2 <= self.x2_light]
            # temp_dist = [norm([x[i] - branches_trimmed2[branch].x2, y[i] - branches_trimmed2[branch].y2]) for
            #              branch in branch_idx]
            #
            # for j in range(0, len(temp_dist)):
            #     if temp_dist[j] < dist:
            #         dist = temp_dist[j]
            #         closest_branch = branch_idx[j]
            #
            # # removes scatter points if reached
            #
            # if dist < mindist:
            #     x = np.delete(x, i)
            #     y = np.delete(y, i)
            #
            # # when distance is greater than max distance, branching occurs to find other points.
            # elif dist < maxdist:
            #     branches_trimmed2[closest_branch].grow_count += 1
            #     branches_trimmed2[closest_branch].grow_x += (x[i] - branches_trimmed2[
            #                                                    closest_branch].x2) / (dist / BRANCH_LENGTH)
            #
            #     branches_trimmed2[closest_branch].grow_y += (y[i] - branches_trimmed2[
            #                                                    closest_branch].y2) / (dist / BRANCH_LENGTH)


        for branch in branches_trimmed:
            if branch.grow_count > 0:
                (x2, y2) = branch.tip_point + branch.grow_direction / branch.grow_count
                x2 = np.clip(x2, 0, self.width-1)
                y2 = np.clip(y2 ,0, self.height -1)
                newBranch = PixelBranch(branch.x2, ir(x2), branch.y2, ir(y2), self.width, self.height)
                self.branches.append(newBranch)
                branch.child.append(newBranch)
                branch.grow_count = 0
                branch.grow_direction.fill(0)


        # increase thickness of first elements added to tree as they grow
        self.branches[0].update_width()
        branch_coords = []

        #sending coordinates out
        for branch in self.branches:
            # x2 and y2 since they are the tips
            branch_coords.append(branch.tip_point)


        self.tips = branch_coords

        for branch in branches_trimmed2:
            if branch.grow_count > 0:
                (x2, y2) = branch.tip_point + branch.grow_direction / branch.grow_count
                x2 = np.clip(x2, 0, self.width-1)
                y2 = np.clip(y2 ,0, self.height -1)
                newBranch2 = PixelBranch(branch.x2, ir(x2), branch.y2, ir(y2), self.width, self.height)
                self.branches2.append(newBranch2)
                branch.child.append(newBranch2)
                branch.grow_count = 0
                branch.grow_direction.fill(0)


        # increase thickness of first elements added to tree as they grow
        self.branches2[0].update_width()
        branch_coords2 = []

        #sending coordinates out
        for branch in self.branches2:
            # x2 and y2 since they are the tips
            branch_coords2.append(branch.tip_point)


        self.tips2 = branch_coords2



        # for i in range(len(branches_trimmed)):
        #     if branches_trimmed[i].grow_count > 0:
        #         newBranch = Branch(
        #             branches_trimmed[i].x2, branches_trimmed[i].x2 +
        #             branches_trimmed[i].grow_x / branches_trimmed[i].grow_count,
        #             branches_trimmed[i].y2, branches_trimmed[i].y2 +
        #             branches_trimmed[i].grow_y / branches_trimmed[i].grow_count,
        #             self.width, self.height)
        #         #self.b_keys.add(i+2)
        #         #print("new branch coords", newBranch.x2, newBranch.y2)
        #         #self.bst.insert([newBranch.x2, newBranch.y2])
        #         self.branches.append(newBranch)
        #         branches_trimmed[i].child.append(newBranch)
        #         branches_trimmed[i].grow_count = 0
        #         branches_trimmed[i].grow_x = 0
        #         branches_trimmed[i].grow_y = 0
        #
        # for i in range(len(branches_trimmed2)):
        #     if branches_trimmed2[i].grow_count > 0:
        #         newBranch = Branch(
        #             branches_trimmed2[i].x2, branches_trimmed2[i].x2 +
        #             branches_trimmed2[i].grow_x / branches_trimmed2[i].grow_count,
        #             branches_trimmed2[i].y2, branches_trimmed2[i].y2 +
        #             branches_trimmed2[i].grow_y / branches_trimmed2[i].grow_count,
        #             self.width, self.height)
        #         #self.b_keys.add(i+2)
        #         #print("new branch coords", newBranch.x2, newBranch.y2)
        #         #self.bst.insert([newBranch.x2, newBranch.y2])
        #         self.branches2.append(newBranch)
        #         branches_trimmed2[i].child.append(newBranch)
        #         branches_trimmed2[i].grow_count = 0
        #         branches_trimmed2[i].grow_x = 0
        #         branches_trimmed2[i].grow_y = 0
        #
        # # increase thickness of first elements added to tree as they grow
        #
        # self.branches[0].update_width()
        #
        # self.branches2[0].update_width()
        # branch_coords = []
        # branch_coords2 = []
        #
        # #sending coordinates out
        # for branch in self.branches:
        #     # x2 and y2 since they are the tips
        #     branch_coords.append([branch.x2, branch.y2])
        #
        # for branch in self.branches2:
        #     # x2 and y2 since they are the tips
        #     branch_coords2.append([branch.x2, branch.y2])
        #
        # #print("branching has occured")

        return branch_coords, branch_coords2

    def distance_target(self, coords):
        # Calculate distance from each tip grown
        dist = distance.cdist(coords, [self.target],
                              'euclidean')
        #dist = norm([coords, self.target])
        # Get smallest distance to target
        min_dist = min(dist)
        #print(min_dist)

        return min_dist

    def get_observation(self, debug_show_scatter=False):
        # new empty image

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.obs_type == 'Binary':

            # ---light beam --- #

            yellow = (0, 128, 128)  # RGB color (dark yellow)
            x1 = ir(self.x1_light * self.width)
            x2 = ir(self.x2_light * self.width)
            cv2.rectangle(
                img, pt1=(x1, 0), pt2=(x2, self.height), color=yellow, thickness=-1)
            # print(img.shape)
            light_img = np.sum(img, axis=2)
            light = np.where(light_img <=128, light_img, 1)

            # ---tree--- #
            img1 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for branch in self.branches:
                pt1, pt2 = branch.get_pt1_pt2()
                thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
                # print("branch width", branch.width, " BRANCH THICCNESS: ", BRANCH_THICCNESS, " width: ", self.width)
                cv2.line(
                    img1,
                    pt1=pt1,
                    pt2=pt2,
                    color=(0, 255, 0),
                    thickness=thiccness)
            tree_img = np.sum(img1, axis=2)
            tree = np.where(tree_img <= 255, tree_img, 1)

            # ---target--- #
            img2 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            x = ir(self.target[0] * self.width)
            y = ir(self.target[1] * self.height)
            cv2.circle(
                img,
                center=(x, y),
                radius=ir(.03 * self.width),
                color=(0, 0, 255),
                thickness=-1)

            target_img = np.sum(img2, axis=2)
            target = np.where(target_img <= 255, target_img, 1)

            final_img = np.dstack((light, tree, target))
            #print("dimensions of binary :",final_img.shape)

            final_img = cv2.flip(final_img, 0)

            return final_img


        if self.obs_type == None:
        # place light as rectangle
            yellow = (0, 128, 128)  # RGB color (dark yellow)
            img[self.feature_maps[Features.light].nonzero()] = yellow

            cv2.rectangle(
                img, pt1=(int(self.x1_light), 0), pt2=(int(self.x2_light),self.height),color=yellow,thickness=-1)

            #
            # x1 = ir(self.x1_light * self.width)
            # x2 = ir(self.x2_light * self.width)
            # cv2.rectangle(
            #     img, pt1=(x1, 0), pt2=(x2, self.height), color=yellow, thickness=-1)
            # # print(f"drawing light rectangle from {(x1, 0)} "
            # #       f"to {(x2, self.height)}")
            # #print(img.shape)

            if debug_show_scatter:
                xs, ys = self.light_scatter()
                for k in range(len(xs)):
                    x = ir(xs[k] * self.width)
                    y = ir(ys[k] * self.height)
                    cv2.circle(
                        img,
                        center=(x, y),
                        radius=2,
                        color=(255, 0, 0),
                        thickness=-1)

            # Draw plant as series of lines (1 branch = 1 line)
            for branch in self.branches:
                thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
                cv2.line(img, pt1=branch.p, pt2=branch.tip_point, color=(0, 255, 0), thickness=thiccness)

                # pt1, pt2 = branch.get_pt1_pt2()
                # thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
                # #print("branch width", branch.width, " BRANCH THICCNESS: ", BRANCH_THICCNESS, " width: ", self.width)
                # cv2.line(
                #     img,
                #     pt1=pt1,
                #     pt2=pt2,
                #     color=(0, 255, 0),
                #     thickness=thiccness)
                # # print(f"drawing branch from {pt1} to {pt2} "
                # #       f"with thiccness {branch.width/50 * self.width}")
            ## draw second tree
            for branch in self.branches2:
                thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
                cv2.line(img, pt1=branch.p, pt2=branch.tip_point, color=(0, 255, 0), thickness=thiccness)

                # pt1, pt2 = branch.get_pt1_pt2()
                # thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
                # #print("branch width", branch.width, " BRANCH THICCNESS: ", BRANCH_THICCNESS, " width: ", self.width)
                # cv2.line(
                #     img,
                #     pt1=pt1,
                #     pt2=pt2,
                #     color=(0, 255, 0),
                #     thickness=thiccness)

            # place goal as filled circle with center and radius
            # also important - place goal last because must be always visible
            x = ir(self.target[0]) #* self.width)
            y = ir(self.target[1])#* self.height)
            cv2.circle(
                img,
                center=(x, y),
                radius=ir(.03 * self.width),
                color=(0, 0, 255),
                thickness=-1)
            # print(f"drawing goal circle at {(x, y)} "
            #       f"with radius {ir(.03*self.width)}")

            # flip image, because plant grows from the bottom, not the top
            img = cv2.flip(img, 0)
            #print(img)

            return img

    def reset(self):
        self.light_width = ir(LIGHT_WIDTH*self.width)

        if self.setting == 'easy':
            random_start = ir(np.random.rand()*self.width)
            random_start2 = ir(random_start + self.light_width)
            self.target = [random_start+(self.light_width/2), 0.8*self.height]

        elif self.setting == 'hard_middle':
            random_start = ir(np.random.uniform(low=0.05, high=0.2)*self.width)
            random_start2 = ir(np.random.uniform(low=0.8, high=0.95)*self.width)

            self.target = [0.5*self.width, 0.8*self.height]

        elif self.setting == 'hard_above':
            coin_flip = np.random.randint(2, size=1)
            random_start = ir(np.random.uniform(low=0.05, high=0.2)*self.width)
            random_start2 = ir(np.random.uniform(low=0.8, high=0.95)*self.width)
            if coin_flip == 0:
                self.target = [random_start, 0.8*self.height]
            if coin_flip == 1:
                self.target = [random_start2, 0.8*self.height]

        else:
            random_start = ir(np.random.rand()*self.width)
            random_start2 = ir(np.random.rand()*self.width)
            self.target = [ir(np.random.rand()*self.width), 0.8*self.height]  # [np.random.uniform(0, 1), .8]

        start_light = ir(np.random.rand()*self.width)
        if np.abs(random_start2-random_start) < self.light_width:
            random_start2 = random_start2 + self.light_width
        if random_start2 > 1*self.width:
            random_start2 = ir(0.99*self.width)

        self.branches = [
            PixelBranch(
                x=random_start,
                x2=random_start,
                y=0,
                y2=FIRST_BRANCH_HEIGHT,
                img_width=self.width,
                img_height=self.height)]

        self.branches2 = [
            PixelBranch(
                x=random_start2,
                x2=random_start2,
                y=0,
                y2=FIRST_BRANCH_HEIGHT,
                img_width=self.width,
                img_height=self.height)]

        if start_light > .87*self.width:
            self.x1_light = .75*self.width
        elif start_light < 0.13*self.width:
            self.x1_light = 0
        else:
            self.x1_light = start_light - (self.light_width / 2)

        self.x2_light = self.x1_light + self.light_width

        #self.x_scatter = np.random.uniform(0, 1, self.light_dif)
        y_scatter = np.random.randint(0, self.width, self.light_dif)
        x_scatter = np.random.randint(FIRST_BRANCH_HEIGHT, self.height, self.light_dif)
        #self.y_scatter = np.random.uniform(FIRST_BRANCH_HEIGHT, 1, self.light_dif)
        self.feature_maps[Features.scatter].fill(False)
        self.feature_maps[Features.scatter][x_scatter, y_scatter] = True
        self.steps = 0
        self.new_branches = 0
        self.tips_per_step = 0
        self.b1 = 0 #branches from plant 1
        self.b2 = 0 #branches from plant 2
        self.light_move = 0
        self.tips = [self.branches[0].tip_point]
        self.tips2 = [self.branches2[0].tip_point]
        self.draw_beam()
        #self.__initialized = True
        return self.get_observation()

    def step(self, action):
        #if not self.__initialized:
            #raise RuntimeError("step() was called before reset()")

        # if action == Actions.move_left:
        #     self.light_move_L()
        #
        # if action == Actions.move_right:
        #     self.light_move_R()
        #
        # if action == Actions.increase_beam:
        #     self.light_increase()
        #
        # if action == Actions.decrease_beam:
        #     self.light_decrease()
        #
        # if action == Actions.noop:
        #     # then we keep the light in place
        #     pass
        #
        if action == 0:
            self.light_move_L()

        if action == 1:
            self.light_move_R()

        if action == 2:
            self.light_increase()

        if action == 3:
            self.light_decrease()

        if action == 4:
            # then we keep the light in place
            pass

        self.x2_light = self.x1_light + self.light_width

        #self.x2_light = self.x1_light + self.light_width
        # filter scattering

        #xs, ys = self.light_scatter()
        pts = self.light_scatter()
        # Branching step for light in this position
        tips = self.tree_grow(pts, .01*self.width, .15*self.height)
        self.draw_beam()
        #print("tips:", tips)
        # Calculate distance to target
        d1 = self.distance_target(tips[0])
        d2 = self.distance_target(tips[1])

        if d1 <= 0.1:
            r1 = (1/0.1 /10) *.5
            #reward = preprocessing.normalize(reward)
        else:
            r1 = (1 / d1 /10) *.5

        if d2 <= 0.1:
            r2 = (1/0.1 /10) *.5
            #reward = preprocessing.normalize(reward)
        else:
            r2 = (1 / d2 /10)*.5

        #reward = r1+r2
        if r1 < r2:
            reward = r1
        elif r2 == r1:
            reward = r2
        else:
            reward = r2

        if reward == 1:
            success = 1
        else:
            success = 0

        #print("this is reward:",reward)
        #reward = preprocessing.normalize(reward)
        # Render image of environment at current state
        observation = self.get_observation()  #image
        #print("these are tips:",tips)
        #print("length of tips:", len(tips))
        plant = (observation[:, :, 1] / 255)  # binary map of plant
        pixel_plant = np.sum(plant)

        done = False  # because we don't have a terminal condition
        misc = {"tips": tips, "target": self.target, "light": self.x1_light, "light_width": self.light_width, "step": self.steps, "success": success}

        if self.steps == 0:
            self.new_branches = len(tips[0]) + len(tips[1])
            self.b1 = len(tips[0])
            self.b2 = len(tips[1])

            misc['new_branches'] = self.new_branches
            misc['new_b1'] = self.b1
            misc['new_b2'] = self.b2
            self.light_move = self.light_move
            #misc['new_branches_1'] = self.new_branches
            #misc['new_branches_2'] = self.new_branches

        else:
            new_branches = len(tips[0])+len(tips[1])-self.new_branches
            new_b1 = len(tips[0]) - self.b1
            new_b2 = len(tips[1]) - self.b2
            misc['new_b1'] = new_b1
            misc['new_b2'] = new_b2
            misc['new_branches'] = new_branches
            self.new_branches = len(tips[0]) + len(tips[1]) # reset for future step
            self.light_move = np.abs(self.light_move - self.x1_light)
            misc['light_move'] = self.light_move

        misc['img'] = observation
        misc['plant_pixel'] = pixel_plant
        # (optional) additional information about plant/episode/other stuff, leave empty for now
        #print("steps:", self.steps)    # sanity check
        self.steps += 1
        #print(misc)
        #self.number_of_branches = new_branches
        #print("how many new branches? ", misc['new_branches'])
        #if done:
            #self.__initialized = True
        return observation, float(reward), done, misc

    def render(self, mode='human',
               debug_show_scatter=False):  # or mode="rgb_array"
        img = self.get_observation(debug_show_scatter)

        if mode == "human":
            cv2.imshow('plant', img)  # create opencv window to show plant
            cv2.waitKey(1)  # this is necessary or the window closes immediately
        else:
            return img

    def draw_beam(self):
        self.feature_maps[Features.light].fill(False)
        cv2.rectangle(
            self.feature_maps[Features.light], pt1=(int(self.x1_light), 0),
            pt2=(int(self.x2_light), self.height),
            color=True ,
            thickness=-1)

if __name__ == '__main__':
    import time

    gse = GrowSpaceEnv_Fairness()

    def key2action(key):
        # if key == ord('a'):
        #     return Actions.move_left
        # elif key == ord('d'):
        #     return Actions.move_right
        # elif key == ord('s'):
        #     return Actions.noop
        # elif key == ord('w'):
        #     return Actions.increase_beam
        # elif key == ord('x'):
        #     return Actions.decrease_beam
        if key == ord('a'):
            return 0 # move left
        elif key == ord('d'):
            return 1 # move right
        elif key == ord('s'):
            return 4 # stay in place
        elif key == ord('w'):
            return 2
        elif key == ord('x'):
            return 3
        else:
            return None

    rewards = []
    while True:
        gse.reset()
        img = gse.get_observation(debug_show_scatter=False)
        cv2.imshow("plant", img)
        rewards = []
        for _ in range(50):
            action = key2action(cv2.waitKey(-1))
            if action is None:
                quit()

            b,t,c,f = gse.step(action)
            #print(f["new_branches"])
            rewards.append(t)
            cv2.imshow("plant", gse.get_observation(debug_show_scatter=False))
        total = sum(rewards)

        print("amount of rewards:", total)