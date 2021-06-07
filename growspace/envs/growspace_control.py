from random import sample

import cv2
import gym
import numpy as np
from numpy.linalg import norm
from growspace.plants.tree import PixelBranch
from scipy.spatial import distance
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import sys
from enum import IntEnum
np.set_printoptions(threshold=sys.maxsize)


BRANCH_THICCNESS = .015

MAX_BRANCHING = 8
DEFAULT_RES = 84
FIRST_BRANCH_HEIGHT = int(.2*DEFAULT_RES)
BRANCH_LENGTH = (1/10)*DEFAULT_RES
LIGHT_WIDTH = .25
LIGHT_DIF = 200
LIGHT_DISPLACEMENT = .1
LIGHT_W_INCREMENT = .1
MIN_LIGHT_WIDTH = .1
MAX_LIGHT_WIDTH = 1

def to_int(v):
    return int(round(v))

def unpack(w):
    return map(list, zip(*enumerate(w)))

def intersection_(coords, light_x):
    xdiff = coords[0] - coords[1]
    ydiff = coords[2] - coords[3]
    delta = ydiff/xdiff
    b = coords[2] - (coords[0] * delta)
    y = (delta * light_x) + b
    return

ir = to_int  # shortcut for function call

class Features(IntEnum):
    light = 0
    scatter = 1

class GrowSpaceEnv_Control(gym.Env):
    def __init__(self, width=DEFAULT_RES, height=DEFAULT_RES, light_dif=LIGHT_DIF, obs_type = None, level = 'second', setting = 'easy'):
        self.width = width
        self.height = height
        self.seed()
        self.light_dif = light_dif
        self.action_space = gym.spaces.Discrete(5)
        self.obs_type = obs_type
        if self.obs_type == None:
            self.observation_space = gym.spaces.Box(
                0, 255, shape=(84, 84, 3), dtype=np.uint8)
        if self.obs_type == 'Binary':
            self.observation_space = gym.spaces.Box(
                0, 1, shape=(84, 84, 3), dtype=np.uint8)
        self.level = level
        self.setting = setting

        self.feature_maps = np.zeros((len(Features), self.height, self.width), dtype=np.uint8)

        self.branches = None
        self.target = None
        self.steps = None
        self.new_branches = None
        self.tips_per_step = None
        self.tips = None

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def light_scatter(self):
        # select scattering with respect to position of the light
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
            self.x1_light = 0  # move by .1

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
            self.light_width = 1*self.width-self.x1_light
        else:
            self.light_width += LIGHT_W_INCREMENT*self.width


    def tree_grow(self, activated_photons, mindist, maxdist):

        branches_trimmed = self.branches
        for i in range(len(activated_photons) - 1, 0, -1):
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

        return branch_coords

    def distance_target(self, coords):
        # Calculate distance from each tip grown
        dist = distance.cdist(coords, [self.target],
                              'euclidean')
        min_dist = min(dist)

        return min_dist

    def get_observation(self, debug_show_scatter=False):
        # new empty image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.obs_type == 'Binary':

            # ---light beam --- #

            yellow = (0, 128 , 128)  # RGB color (dark yellow)
            x1 = ir(self.x1_light * self.width)
            x2 = ir(self.x2_light * self.width)
            cv2.rectangle(
                img, pt1=(x1, 0), pt2=(x2, self.height), color=yellow, thickness=-1)
            light_img = np.sum(img, axis=2)
            light = np.where(light_img <=128, light_img, 1)

            # ---tree--- #
            img1 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for branch in self.branches:
                pt1, pt2 = branch.get_pt1_pt2()
                thiccness = ir(branch.width * BRANCH_THICCNESS * self.width)
                cv2.line(
                    img1,
                    pt1=pt1,
                    pt2=pt2,
                    color=(0, 255, 0),
                    thickness=thiccness)
            tree_img = np.sum(img1, axis=2)
            tree = np.where(tree_img < 255, tree_img, 1)

            # ---light + tree ----#
            light_tree = light+tree  #addition of matrices
            light_tree_binary = np.where(light_tree < 2,light_tree, 1)
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
            target = np.where(target_img < 255, target_img, 1)
            light_target = light + target  # additions of matrices
            light_target_binary = np.where(light_target< 2, light_target, 1)
            final_img = np.dstack((light, light_tree_binary, light_target_binary))
            final_img = cv2.flip(final_img, 0)

            return final_img


        if self.obs_type == None:
            # place light as rectangle
            yellow = (0, 128, 128)  # RGB color (dark yellow)

            img[self.feature_maps[Features.light].nonzero()] = yellow
            ## I am here now april 21

            cv2.rectangle(
                img, pt1=(int(self.x1_light), 0), pt2=(int(self.x2_light), self.height), color=yellow, thickness=-1)


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


            x = ir(self.target[0])
            y = ir(self.target[1])
            cv2.circle(
                img,
                center=(x, y),
                radius=ir(.03 * self.width),
                color=(0, 0, 255),
                thickness=-1)

            img = cv2.flip(img, 0)

            return img

    def reset(self):
        # Set env back to start - also necessary on first start
        # is in range [0,1]
        if self.setting == 'easy':
            random_start = ir(np.random.rand()*self.width)
            random_start2 = random_start
            self.target = [random_start, .8*self.height]

        elif self.setting == 'hard':
            coin_flip = np.random.randint(2, size = 1)
            if coin_flip == 0:
                random_start = ir(np.random.uniform(low = 0.05, high = 0.2)* self.width)
                random_start2 = ir(np.random.uniform(low = 0.8, high = 0.95)*self.width)
            if coin_flip == 1:
                random_start = ir(np.random.uniform(low = 0.8, high = 0.95)*self.width)
                random_start2 = ir(np.random.uniform(low = 0.05, high = 0.2)*self.width)

            self.target = [random_start, .8*self.height]
        else:
            random_start = ir(np.random.rand()*self.width) # is in range [0,1]
            random_start2 = ir(np.random.rand()*self.width)
            self.target = [random_start, .8*self.height]

        self.branches = [
            PixelBranch(
                x=random_start2,
                x2=random_start2,
                y=0,
                y2=FIRST_BRANCH_HEIGHT,
                img_width=self.width,
                img_height=self.height)]


        self.light_width = ir(.25*self.width)
        if self.level == None:
            start_light = random_start2
            #self.x1_light = random_start - (self.light_width/2)

        if self.level == 'second':
            if self.setting == 'hard':
                start_light = self.target[0]
            elif self.setting == 'easy':
                start_light = ir(np.random.rand()*self.width)
            else:
                start_light = ir(np.random.rand()*self.width)

        if start_light > ir(self.width - self.light_width/2):
            start_light = ir(self.width - self.light_width /2)
            self.x1_light = start_light - (self.light_width / 2)

        elif start_light < ir(self.light_width):
            start_light = ir(self.light_width/2)
            self.x1_light = start_light - (self.light_width / 2)

        else:
            self.x1_light = start_light - (self.light_width / 2)

        self.x2_light = self.x1_light + self.light_width

        self.light_coords = [self.x1_light, self.x2_light]
        y_scatter = np.random.randint(0,self.width, self.light_dif)
        x_scatter = np.random.randint(FIRST_BRANCH_HEIGHT, self.height, self.light_dif)
        self.feature_maps[Features.scatter].fill(False)
        self.feature_maps[Features.scatter][x_scatter, y_scatter] = True
        self.steps = 0
        self.new_branches = 0
        self.tips_per_step = 0
        self.light_move = 0
        #self.tips = [self.branches[0].x2, self.branches[0].y2]
        self.tips = [self.branches[0].tip_point]
        self.draw_beam()

        return self.get_observation()

    def step(self, action):
        # Two possible actions, move light left or right

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


        pts = self.light_scatter()

        tips = self.tree_grow(pts, .01 * self.width, .15 * self.height)
        self.draw_beam()

        # Calculate distance to target

        if self.distance_target(tips) <= 3.2: # before was 0.1
            reward = 1/3.2 #0.1 /10
            success = 1

        else:
            reward = 1 / self.distance_target(tips) #adsss/10
            success = 0

        # Render image of environment at current state
        observation = self.get_observation()  #image
        plant = (observation[:,:,1]/255) # binary map of plant
        pixel_plant = np.sum(plant)

        done = False  # because we don't have a terminal condition
        misc = {"tips": tips, "target": self.target, "light": self.x1_light, "light_width": self.light_width, "step": self.steps, "success": success }

        if self.steps == 0:
            self.new_branches = len(tips)
            misc['new_branches'] = self.new_branches
            self.light_move = self.light_move

        else:
            new_branches = len(tips)-self.new_branches
            misc['new_branches'] = new_branches
            self.new_branches = len(tips)  # reset for future step
            self.light_move = np.abs(self.light_move - self.x1_light)
            misc['light_move'] = self.light_move

        misc['img'] = observation
        misc["plant_pixel"] = pixel_plant

        # (optional) additional information about plant/episode/other stuff, leave empty for now

        self.steps += 1

        return observation, float(reward), done, misc

    def render(self, mode='human',
               debug_show_scatter=False):  # or mode="rgb_array"
        img = self.get_observation(debug_show_scatter)

        if self.obs_type == 'Binary':
            image = img.astype(np.uint8)
            img = image * 255

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

    def to_image(self, p):
        if hasattr(p, "normalized_array"):
            return np.around((self.height, self.width) * p.normalized_array[:-1]).astype(np.int32)
        else:
            y, x = p
            return np.around((self.height * y, self.width * x)).astype(np.int32)


if __name__ == '__main__':
    import time

    gse = GrowSpaceEnv_Control()

    def key2action(key):
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
    rewards_mean = []
    while True:
        gse.reset()
        img = gse.get_observation(debug_show_scatter=False)

        cv2.imshow("plant", img)
        rewards = []
        for _ in range(70):
            action = key2action(cv2.waitKey(-1))
            if action is None:
                quit()

            b,t,c,f = gse.step(action)
            #print(f["new_branches"])
            rewards.append(t)
            #print(t)
            cv2.imshow("plant", gse.get_observation(debug_show_scatter=False))
        total = sum(rewards)

        rewards_mean.append(total)
        av = np.mean(rewards_mean)
        print("amount of rewards:", total)
        print('mean:', av)
        print("amount of rewards:", total)
