from random import sample

import cv2
import gym
import numpy as np
from numpy.linalg import norm
import time
from growspace.plants.tree import Branch
from scipy.spatial import distance
#import sys
#np.set_printoptions(threshold=sys.maxsize)
import itertools
from sklearn import preprocessing


from numba import jit
from functools import partial



FIRST_BRANCH_HEIGHT = .24
BRANCH_THICCNESS = .015
BRANCH_LENGTH = 1/9
MAX_BRANCHING = 10

def to_int(v):
    return int(round(v))

def unpack(w):
    return map(list, zip(*enumerate(w)))


ir = to_int  # shortcut for function call


class GrowSpaceEnv(gym.Env):

    def __init__(self, width=84, height=84, light_dif=250, obs_type = None, level= None):
        self.width = width  # do we keep?
        self.height = height  # do we keep?
        self.seed()
        self.light_dif = light_dif
        self.action_space = gym.spaces.Discrete(3)  # L, R, keep of light paddle
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self.obs_type = obs_type
        self.level = level
        # note: I moved the code for the first branch into the reset function,
        # because when you start an environment for the first time,
        # you're supposed to call "reset()" first before doing anything else

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def light_scatter(self):
        # select scattering with respect to position of the light

        # select the instances where the conditions is true (where the x coordinate is within the light)
        filter = np.logical_and(self.x_scatter >= self.x1_light,
                                self.x_scatter <= self.x2_light)

        # apply filter to both y and x coordinates through the power of Numpy magic :D
        ys = self.y_scatter[filter]
        xs = self.x_scatter[filter]
        return xs, ys

    def light_move_R(self):
        if np.around(self.x1_light,1) >= .8:  # limit of coordinates
            self.x1_light = .8  # stay put

        else:
            self.x1_light += .1  # move by .1 right

    def light_move_L(self):
        if np.around(self.x1_light,1) <= .1:  # limit of coordinates
            self.x1_light = 0
        else:
            self.x1_light -= .1  # move by .1 left


    #@staticmethod
    #@jit(nopython=True)
    #@partial(jit, static_argnums=(0,))
    def tree_grow(self,x, y, mindist, maxdist):

        # apply filter to both idx and branches
        for i in range(len(x) - 1, 0, -1):  # number of possible scatters, check if they allow for branching with min_dist
            closest_branch = 0
            dist = 1

            if len(self.branches) > MAX_BRANCHING:
                branches_trimmed = sample(self.branches, MAX_BRANCHING)
            else:
                branches_trimmed = self.branches
            branch_idx = [branch_idx for branch_idx, branch in enumerate(branches_trimmed)if self.x1_light <= branch.x2 <= self.x2_light]
            temp_dist = [norm([x[i] - branches_trimmed[branch].x2, y[i] - branches_trimmed[branch].y2]) for branch in branch_idx]

            for j in range(0, len(temp_dist)):
                if temp_dist[j] < dist:
                    dist = temp_dist[j]
                    closest_branch = branch_idx[j]


            # removes scatter points if reached

            if dist < mindist:
                x = np.delete(x, i)
                y = np.delete(y, i)

            # when distance is greater than max distance, branching occurs to find other points.
            elif dist < maxdist:

                #self.branches[closest_branch].grow_count += 1
                branches_trimmed[closest_branch].grow_count += 1
                #self.branches[closest_branch].grow_x += (
                    #x[i] - self.branches[closest_branch].x2) / (dist/BRANCH_LENGTH)
                branches_trimmed[closest_branch].grow_x += (
                    x[i] - branches_trimmed[closest_branch].x2) / (dist/BRANCH_LENGTH)
                #self.branches[closest_branch].grow_y += (
                    #y[i] - self.branches[closest_branch].y2) / (dist/BRANCH_LENGTH)
                branches_trimmed[closest_branch].grow_y += (
                    y[i] - branches_trimmed[closest_branch].y2) / (dist / BRANCH_LENGTH)


        for i in range(len(branches_trimmed)):
            if branches_trimmed[i].grow_count > 0:
                newBranch = Branch(
                    branches_trimmed[i].x2, branches_trimmed[i].x2 +
                    branches_trimmed[i].grow_x / branches_trimmed[i].grow_count,
                    branches_trimmed[i].y2, branches_trimmed[i].y2 +
                    branches_trimmed[i].grow_y / branches_trimmed[i].grow_count,
                    self.width, self.height)
                #self.b_keys.add(i+2)
                #print("new branch coords", newBranch.x2, newBranch.y2)
                #self.bst.insert([newBranch.x2, newBranch.y2])
                self.branches.append(newBranch)
                branches_trimmed[i].child.append(newBranch)
                branches_trimmed[i].grow_count = 0
                branches_trimmed[i].grow_x = 0
                branches_trimmed[i].grow_y = 0

        # increase thickness of first elements added to tree as they grow

        self.branches[0].update_width()


        branch_coords = []

        #sending coordinates out
        for branch in self.branches:
            # x2 and y2 since they are the tips
            branch_coords.append([branch.x2, branch.y2])

        #print("branching has occured")

        return branch_coords

    def distance_target(self, coords):
        # Calculate distance from each tip grown
        dist = distance.cdist(coords, [self.target],
                              'euclidean')  #TODO replace with numpy
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
            #print("light img to understand",light_img)
            light = np.where(light_img <=128, light_img, 1)
            #print("this is light", light)
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
            #print("this is tree_img", tree_img)
            tree = np.where(tree_img < 255, tree_img, 1)
            #print("this is tree",tree)

            # ---target--- #
            img2 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            x = ir(self.target[0] * self.width)
            y = ir(self.target[1] * self.height)
            cv2.circle(
                img2,
                center=(x, y),
                radius=ir(.03 * self.width),
                color=(0, 0, 255),
                thickness=-1)

            target_img = np.sum(img2, axis=2)
            #print("test",target_img)
            target = np.where(target_img < 255, target_img, 1)
            #print("this is a target", target)
            final_img = np.dstack((light, tree, target))
            #print("dimensions of binary :",final_img.shape)

            final_img = cv2.flip(final_img, 0)

            return final_img


        if self.obs_type == None:
        # place light as rectangle
            yellow = (0, 128, 128)  # RGB color (dark yellow)
            x1 = ir(self.x1_light * self.width)
            x2 = ir(self.x2_light * self.width)
            cv2.rectangle(
                img, pt1=(x1, 0), pt2=(x2, self.height), color=yellow, thickness=-1)
            # print(f"drawing light rectangle from {(x1, 0)} "
            #       f"to {(x2, self.height)}")
            #print(img.shape)

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
                pt1, pt2 = branch.get_pt1_pt2()
                thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
                #print("branch width", branch.width, " BRANCH THICCNESS: ", BRANCH_THICCNESS, " width: ", self.width)
                cv2.line(
                    img,
                    pt1=pt1,
                    pt2=pt2,
                    color=(0, 255, 0),
                    thickness=thiccness)
                # print(f"drawing branch from {pt1} to {pt2} "
                #       f"with thiccness {branch.width/50 * self.width}")


            # place goal as filled circle with center and radius
            # also important - place goal last because must be always visible
            x = ir(self.target[0] * self.width)
            y = ir(self.target[1] * self.height)
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
        # Set env back to start - also necessary on first start
        random_start = np.random.rand()  # is in range [0,1]
        #random_start = 0.01
        self.branches = [
            Branch(
                x=random_start,
                x2=random_start,
                y=0,
                y2=FIRST_BRANCH_HEIGHT,
                img_width=self.width,
                img_height=self.height)
        ]
        #self.target = [np.random.uniform(0, 1), np.random.uniform(.8, 1)]
        self.target = [np.random.uniform(0, 1), .8]
        #self.target = [0.01, 0.01]
        self.light_width = .25
        if random_start > .87:
            self.x1_light = .75
        elif random_start < 0.13:
            self.x1_light = 0
        else:
            self.x1_light = random_start - (self.light_width/2)

        if self.level == 'second':
            start_light = np.random.rand()
            if start_light > .87:
                self.x1_light = .75
            elif start_light < 0.13:
                self.x1_light = 0
            else:
                self.x1_light = start_light - (self.light_width / 2)

        self.x2_light = self.x1_light + self.light_width

        self.x_scatter = np.random.uniform(0, 1, self.light_dif)
        self.y_scatter = np.random.uniform(0.25, 1, self.light_dif)
        self.steps = 0
        self.new_branches = 0
        self.tips_per_step = 0

        return self.get_observation()

    def step(self, action):
        # Two possible actions, move light left or right

        if action == 0:
            self.light_move_L()

        if action == 1:
            self.light_move_R()

        self.x2_light = self.x1_light + self.light_width

        if action == 2:
            # then we keep the light in place
            pass


        # filter scattering
        xs, ys = self.light_scatter()

        # Branching step for light in this position
        tips = self.tree_grow(xs, ys, .01, .15)
        #print("tips:", tips)
        # Calculate distance to target
        if self.distance_target(tips) <= 0.1:
            reward = 1/0.1 /10
            #reward = preprocessing.normalize(reward)
        else:
            reward = 1 / self.distance_target(tips) /10
        #print("this is reward:",reward)
        #reward = preprocessing.normalize(reward)
        # Render image of environment at current state
        observation = self.get_observation()  #image
        #print("these are tips:",tips)
        #print("length of tips:", len(tips))

        done = False  # because we don't have a terminal condition
        misc = {"tips": tips, "target": self.target, "light": self.x1_light}

        if self.steps == 0:
            self.new_branches = len(tips)
            misc['new_branches'] = self.new_branches

        else:
            new_branches = len(tips)-self.new_branches
            misc['new_branches'] = new_branches
            self.new_branches = len(tips)  # reset for future step

        misc['img'] = observation
        # (optional) additional information about plant/episode/other stuff, leave empty for now
        #print("steps:", self.steps)    # sanity check
        self.steps += 1
        #print(misc)
        #self.number_of_branches = new_branches
        #print("how many new branches? ", misc['new_branches'])
        return observation, reward, done, misc

    def render(self, mode='human',
               debug_show_scatter=False):  # or mode="rgb_array"
        img = self.get_observation(debug_show_scatter)

        if mode == "human":
            cv2.imshow('plant', img)  # create opencv window to show plant
            cv2.waitKey(1)  # this is necessary or the window closes immediately
        else:
            return img


if __name__ == '__main__':
    import time

    gse = GrowSpaceEnv()

    def key2action(key):
        if key == ord('a'):
            return 0 # move left
        elif key == ord('d'):
            return 1 # move right
        elif key == ord('s'):
            return 2 # stay in place
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
            print(f["new_branches"])
            rewards.append(t)
            cv2.imshow("plant", gse.get_observation(debug_show_scatter=False))
        total = sum(rewards)

        print("amount of rewards:", total)
