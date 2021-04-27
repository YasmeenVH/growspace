from random import sample

import cv2
import gym
import numpy as np
from numpy.linalg import norm
from growspace.plants.tree import Branch
from scipy.spatial import distance
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import sys
np.set_printoptions(threshold=sys.maxsize)

FIRST_BRANCH_HEIGHT = .2
BRANCH_THICCNESS = .015
BRANCH_LENGTH = 1/10
MAX_BRANCHING = 8
DEFAULT_RES = 84
LIGHT_WIDTH = .25
LIGHT_DIF = 200
LIGHT_DISPLACEMENT = .1
LIGHT_W_INCREMENT = .1
MIN_LIGHT_WIDTH = .1
MAX_LIGHT_WIDTH = 1
"""
import config
for k in list(locals()):
    if f"^" + k in config.tensorboard.run.config:
        locals()[k] = config.tensorboard.run.config[f"^" + k]
"""


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

class GrowSpaceEnv_Control(gym.Env):
    def __init__(self, width=DEFAULT_RES, height=DEFAULT_RES, light_dif=LIGHT_DIF, obs_type = None, level = None, setting = 'easy'):
        self.width = width  # do we keep?
        self.height = height  # do we keep?
        self.seed()
        self.light_dif = light_dif
        self.action_space = gym.spaces.Discrete(5)  # L, R, keep of light paddle
        self.obs_type = obs_type
        if self.obs_type == None:
            self.observation_space = gym.spaces.Box(
                0, 255, shape=(84, 84, 3), dtype=np.uint8)
        if self.obs_type == 'Binary':
            self.observation_space = gym.spaces.Box(
                0, 1, shape=(84, 84, 3), dtype=np.uint8)
        self.level = level
        self.setting = setting

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
        #print("scatter:", xs, ys)
        return xs, ys

    def light_move_R(self):
        if np.around(self.x1_light + self.light_width,2) <= 1 - LIGHT_DISPLACEMENT:  # limit of coordinates
            self.x1_light += LIGHT_DISPLACEMENT  # stay put
        else:
            self.x1_light = 1 - self.light_width
            #self.x1_light += .1  # move by .1 right

    def light_move_L(self):
        if np.around(self.x1_light,2) >= LIGHT_DISPLACEMENT:  # limit of coordinates
            #print("what is happening???", self.x1_light)
            self.x1_light -= LIGHT_DISPLACEMENT
        else:
            self.x1_light = 0  # move by .1 leftdd

    def light_decrease(self):
        if np.around(self.light_width,1) <= MIN_LIGHT_WIDTH:
            self.light_width = self.light_width
            #passd
       # elif MIN_LIGHT_WIDTH < self.light_width < MIN_LIGHT_WIDTH + LIGHT_W_INCREMENT/2:
            #self.light_width = self.light_width
        else:
            self.light_width -= LIGHT_W_INCREMENT

    def light_increase(self):
        if self.light_width >= MAX_LIGHT_WIDTH:
            #self.light_width = self.light_width
            pass
        elif self.x1_light + self.light_width >= 1:
            self.light_width = 1-self.x1_light
        else:
            self.light_width += LIGHT_W_INCREMENT

    #@staticmethod
    #@jit(nopython=True)
    #@partial(jit, static_argnums=(0,))
    def tree_grow(self,x, y, mindist, maxdist):

        # apply filter to both idx and branches
        #print("what is value",len(x))

        #if len(x) == None:
            #print("pass")
            #branches_trimmed = [0]
            #pass
       # elif len(x) == 1:
            #closest_branch = 0
            #dist = 1
            #if len(self.branches) > MAX_BRANCHING:
                #branches_trimmed = sample(self.branches, MAX_BRANCHING)
            #else:
                #branches_trimmed = self.branches
            #branch_idx = [branch_idx for branch_idx, branch in enumerate(branches_trimmed) if
                         # self.x1_light <= branch.x2 <= self.x2_light]
           # temp_dist = [norm([x - branches_trimmed[branch].x2, y - branches_trimmed[branch].y2]) for branch in
                         #branch_idx]

            #for j in range(0, len(temp_dist)):
             #   if temp_dist[j] < dist:
             #       dist = temp_dist[j]
             #       closest_branch = branch_idx[j]

            # removes scatter points if reached

            #if dist < mindist:
             #   x = np.delete(x)
            #y = np.delete(y)
#
            # when distance is greater than max distance, branching occurs to find other points.
            #elif dist < maxdist:
                #branches_trimmed[closest_branch].grow_count += 1
                #branches_trimmed[closest_branch].grow_x += (x - branches_trimmed[closest_branch].x2) / (dist / BRANCH_LENGTH)
                #branches_trimmed[closest_branch].grow_y += (y - branches_trimmed[closest_branch].y2) / (dist / BRANCH_LENGTH)
        #else:
        global branches_trimmed
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
                branches_trimmed[closest_branch].grow_count += 1
                branches_trimmed[closest_branch].grow_x += (
                    x[i] - branches_trimmed[closest_branch].x2) / (dist/BRANCH_LENGTH)
                branches_trimmed[closest_branch].grow_y += (
                    y[i] - branches_trimmed[closest_branch].y2) / (dist / BRANCH_LENGTH)

        #print('branches trimmed', branches_trimmed)
        #if branches_trimmed == 0:
            #pass
        for i in range(len(branches_trimmed)):
            if branches_trimmed[i].grow_count > 0:
                newBranch = Branch(
                    branches_trimmed[i].x2, branches_trimmed[i].x2 +
                    branches_trimmed[i].grow_x / branches_trimmed[i].grow_count,
                    branches_trimmed[i].y2, branches_trimmed[i].y2 +
                    branches_trimmed[i].grow_y / branches_trimmed[i].grow_count,
                    self.width, self.height)
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
            #print("dimensions of final shape", np.shape(final_img))
            return final_img


        if self.obs_type == None:
            # place light as rectangle
            yellow = (0, 128, 128)  # RGB color (dark yellow)
            x1 = ir(self.x1_light * self.width)
            x2 = ir(self.x2_light * self.width)
            cv2.rectangle(
                img, pt1=(x1, 0), pt2=(x2, self.height), color=yellow, thickness=-1)

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
                cv2.line(
                    img,
                    pt1=pt1,
                    pt2=pt2,
                    color=(0, 255, 0),
                    thickness=thiccness)

            x = ir(self.target[0] * self.width)
            y = ir(self.target[1] * self.height)
            cv2.circle(
                img,
                center=(x, y),
                radius=ir(.03 * self.width),
                color=(0, 0, 255),
                thickness=-1)

            img = cv2.flip(img, 0)

            #print("dimensions of final shape", np.shape(img))
            return img

    def reset(self):
        # Set env back to start - also necessary on first start
        # is in range [0,1]
        if self.setting == 'easy':
            random_start = np.random.rand()
            random_start2 = random_start
            self.target = [random_start, .8]

        elif self.setting == 'hard':
            coin_flip = np.random.randint(2, size = 1)
            if coin_flip == 0:
                random_start = np.random.uniform(low = 0.05, high = 0.2)
                random_start2 = np.random.uniform(low = 0.8, high = 0.95)
            if coin_flip == 1:
                random_start = np.random.uniform(low = 0.8, high = 0.95)
                random_start2 = np.random.uniform(low = 0.05, high = 0.2)

            self.target = [random_start, .8]
        else:
            random_start = np.random.rand() # is in range [0,1]
            random_start2 = np.random.rand()
            self.target = [random_start, .8]

        self.branches = [
            Branch(
                x=random_start2,
                x2=random_start2,
                y=0,
                y2=FIRST_BRANCH_HEIGHT,
                img_width=self.width,
                img_height=self.height)]

        #self.target = [np.random.uniform(0, 1), np.random.uniform(.8, 1)]
        #self.target = [np.random.uniform(0, 1), .8]
        self.light_width = .25
        if self.level == None:
            start_light = random_start2
            #self.x1_light = random_start - (self.light_width/2)

        if self.level == 'second':
            if self.setting == 'hard':
                start_light = self.target[0]
            elif self.setting == 'easy':
                start_light = np.random.rand()
            else:
                start_light = np.random.rand()

        if start_light > .87:
            self.x1_light = .75
        elif start_light < 0.13:
            self.x1_light = 0
        else:
            self.x1_light = start_light - (self.light_width / 2)

        self.x2_light = self.x1_light + self.light_width

        self.x_scatter = np.random.uniform(0, 1, self.light_dif)
        self.y_scatter = np.random.uniform(FIRST_BRANCH_HEIGHT, 1, self.light_dif)
        self.steps = 0
        self.new_branches = 0
        self.tips_per_step = 0
        self.light_move = 0
        self.tips = [self.branches[0].x2, self.branches[0].y2]

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
        #print("light x1", self.x1_light)
        #print('action',action)
        self.x2_light = self.x1_light + self.light_width

        # filter scattering
        # try:
        #     convex_tips = np.array(self.tips)
            #print('what is length:',len(convex_tips))
        xs, ys = self.light_scatter()
        #     if len(convex_tips) >= 2:
        #         #print('check')
        #         hull = ConvexHull(convex_tips)
        #         #print('check2')
        #         #print('what is this:',convex_tips[hull.vertices,0])
        #        # print('what is this8:', convex_tips[hull.vertices, 1])
        #         xxs = convex_tips[hull.vertices, 0]  # x coords for convex hull around tips
        #         yys = convex_tips[hull.vertices, 1]  # y coords for convex hull around tips
        #         x_max_idx = np.where(xxs == np.amax(xxs))  # idx where most right tip is
        #         x_min_idx = np.where(xxs == np.amin(xxs))  # idx where most left tip is
        #         y_max_idx = np.where(yys == np.amax(yys))  # idx highest tip
        #         #print(x_max_idx[0], 'this is the idx')
        #         #x_min_idx = np.where(min(xxs))
        #         y_max = xxs[y_max_idx]
        #         #print(xs,'this is xs')
        #
        #         if self.x1_light < xxs[x_min_idx] < self.x2_light:  # if within the horizontal beam
        #
        #             if self.x1_light < xxs[x_max_idx] < self.x2_light:
        #
        #                 # full convex is in within the beam
        #                 if yys[x_min_idx] < yys[x_max_idx]:
        #                     if xxs[x_min_idx] < xxs[x_max_idx]:
        #                         filter1 = np.logical_and(xs >= xxs[x_min_idx], xs <= xxs[x_max_idx])
        #                         filter2 = np.logical_and(ys >= 0, ys <= yys[x_min_idx])
        #                     else:
        #                         filter1 = np.logical_and(xs >= xxs[x_max_idx], xs <= xxs[x_min_idx])
        #                         filter2 = np.logical_and(ys >= 0, ys <= yys[x_max_idx])
        #
        #             # convex is on right side and in between beam
        #             filter1 = np.logical_and(xs >= xxs[x_min_idx], xs <= self.x2_light)
        #             filter2 = np.logical_and(ys <= yys[x_min_idx], ys >= 0)
        #
        #             # Filtering for values that are not shaded
        #             idx = [i for i in range(len(filter1)) if (filter1[i] == False) and (filter2[i] == False)]
        #             xs = [xs[i] for i in idx]
        #             ys = [ys[i] for i in idx]
        #
        #         elif self.x1_light < xxs[x_max_idx] < self.x2_light:
        #
        #             # convex is on left side and in between beam
        #             filter1 = np.logical_and(xs <= xxs[x_max_idx], xs >= self.x1_light)
        #             filter2 = np.logical_and(ys >= 0,ys <= yys[x_max_idx])
        #
        #             # Filtering for values that are not shaded
        #             idx = [i for i in range(len(filter1)) if (filter1[i] == False) and (filter2[i] == False)]
        #             xs = [xs[i] for i in idx]
        #             ys = [ys[i] for i in idx]
        #
        #         elif xxs[x_min_idx] < self.x1_light < self.x2_light < xxs[x_max_idx]:
        #             # width of light is covering convex with no exposed edges
        #             intersection_1 = []
        #             intersection_2 = []
        #
        #             for i in range(len(xxs)):
        #                 # find intersecting vertex with convex
        #                 if xxs[i+1]< self.x1_light < xxs[i]:
        #                     intersection_1.append(xxs[i], xxs[i+1])
        #                     intersection_1.append(yys[i], yys[i+1])
        #
        #                 if xxs[i+1] < self.x2_light < xxs[i]:
        #                     intersection_2.append(xxs[i], xxs[i+1])
        #                     intersection_2.append(yys[i], yys[i+1])
        #                     #print('what is intersection2',intersection_2)
        #             y1 = intersection_(intersection_1,self.x1_light)
        #             y2 = intersection_(intersection_2,self.x2_light)
        #             y_limit = min(y1,y2)
        #             filter1 = np.logical_and(xs <= self.x1_light, xs >= self.x2_light)
        #             filter2 = np.logical_and(ys >= 0, ys <= y_limit)
        #
        #             idx = [i for i in range(len(filter1)) if (filter1[i] == False) and (filter2[i] == False)]
        #             #print("what is idx", idx)
        #             xs = [xs[i] for i in idx]
        #             ys = [ys[i] for i in idx]
        #
        #         else:
        #             pass #this  is when beam is not covering plants
        # except:
        #     pass


        #print("scattering x len:", len(xs))
        #print("this is lightx1 :", self.x1_light, "and light width:", self.light_width)
        # Branching step for light in this position
        tips = self.tree_grow(xs, ys, .01, .15)
        #print("tips:", tips)
        # Calculate distance to target
        if self.distance_target(tips) <= 0.1:
            reward = 1/0.1 /10
            success = 1
            #reward = preprocessing.normalize(reward)
        else:
            reward = 1 / self.distance_target(tips) /10
            success = 0
        #print("this is reward:",reward)
        #reward = preprocessing.normalize(reward)
        # Render image of environment at current state
        observation = self.get_observation()  #image
        #print("these are tips:",tips)
        #print("length of tips:", len(tips))

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
        #print(self.light_width, "this is light width")
        # (optional) additional information about plant/episode/other stuff, leave empty for now
        #print("steps:", self.steps)    # sanity check
        self.steps += 1
        #print(misc)
        #self.number_of_branches = new_branches
        #print("how many new branches? ", misc['new_branches'])
        #print("what type of data", type(misc['new_branches']))
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
    while True:
        gse.reset()
        img = gse.get_observation(debug_show_scatter=False)
        #image = img.astype(np.uint8)
        #backtorgb = image * 255
        #wwasssssssssssssssssssssssssssssssssssssssprint(backtorgb)
        cv2.imshow("plant", img)
        rewards = []
        for _ in range(70):
            action = key2action(cv2.waitKey(-1))
            if action is None:
                quit()

            b,t,c,f = gse.step(action)
            print(f["new_branches"])
            rewards.append(t)
            cv2.imshow("plant", gse.get_observation(debug_show_scatter=False))
        total = sum(rewards)

        print("amount of rewards:", total)