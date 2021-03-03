from random import sample
import cv2
import gym
import numpy as np
from numpy.linalg import norm
import time
from growspace.plants.tree import Branch
from scipy.spatial import distance
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.stats import linregress
import sys
np.set_printoptions(threshold=sys.maxsize)


#from numba import jit
#from functools import partial

# customizable variables by user

FIRST_BRANCH_HEIGHT = .24
BRANCH_THICCNESS = .015
BRANCH_LENGTH = 1/9
MAX_BRANCHING = 10
DEFAULT_RES = 71
LIGHT_WIDTH = .25
LIGHT_DIF = 250
LIGHT_DISPLACEMENT = .1
LIGHT_W_INCREMENT = .1
MIN_LIGHT_WIDTH = .1
MAX_LIGHT_WIDTH = .5
PATH = '/home/y/Documents/finalprojectcomp767/growspace/scripts/png/mnist_data/mnist_1.png'

def to_int(v):
    return int(round(v))

def unpack(w):
    return map(list, zip(*enumerate(w)))

ir = to_int  # shortcut for function calld


class GrowSpaceEnv_Mnist(gym.Env):

    def __init__(self, width=DEFAULT_RES, height=DEFAULT_RES, obs_type = None, level=None, path=PATH):
        self.width = width  # do we keep?
        self.height = height  # do we keep?
        self.seed()
        self.action_space = gym.spaces.Discrete(5)  # L, R, keep of light paddle, or increase, decrease
        self.obs_type = obs_type
        if self.obs_type == None:
            self.observation_space = gym.spaces.Box(
                0, 255, shape=(28, 28, 3), dtype=np.uint8)
        if self.obs_type == 'Binary':
            self.observation_space = gym.spaces.Box(
                0, 1, shape=(84, 84, 5), dtype=np.uint8)

        self.level = level
        self.mnist_shape = cv2.imread(path)


    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def light_scatter(self):
        # select scattering with respect to position of the light

        # select the instances where the conditions is true (where the x coordinate is within the light)

        if self.x2_light <= LIGHT_DISPLACEMENT or self.x1_light >= 1-LIGHT_WIDTH:    # either 0 or 1 otherwise never the same
                filter = np.logical_and(self.y_scatter <= self.y1_light,
                                self.y_scatter >= self.y2_light)
        else:
            filter = np.logical_and(self.x_scatter >= self.x1_light,
                                self.x_scatter <= self.x2_light)


        # apply filter to both y and x coordinates through the power of Numpy magic :D
        ys = self.y_scatter[filter]
        xs = self.x_scatter[filter]
        #print(xs, ys)
        return xs, ys

    def light_move_R(self):

        if np.around(self.x1_light, 2) >= 1 - self.light_width:  # limit of coordinates
            if self.y1_light <= FIRST_BRANCH_HEIGHT + self.light_width:
                self.y1_light = FIRST_BRANCH_HEIGHT + self.light_width  #stay put in y axis
            else:
                self.y1_light -= LIGHT_DISPLACEMENT
                self.y2_light = self.y1_light - self.light_width

        elif np.around(self.x2_light,2) <= LIGHT_DISPLACEMENT:  #  Indicator that we are Left
            #if 1-LIGHT_DISPLACEMENT <= self.y1_light <= 1:
            if np.around(self.y1_light,2) > 1-LIGHT_DISPLACEMENT:
                    self.x1_light = 0
                    self.x2_light = self.x1_light + self.light_width
            else:
                self.y1_light += LIGHT_DISPLACEMENT
                self.y2_light = self.y1_light - self.light_width

        else:
            self.x1_light += LIGHT_DISPLACEMENT  # move by defined amount of pixels
            self.x2_light = self.x1_light + self.light_width


    def light_move_L(self):

        if np.around(self.x2_light,2) <= self.light_width: #LIGHT_DISPLACEMENT:  # limit of coordinates

            if self.y1_light <= FIRST_BRANCH_HEIGHT + self.light_width:
                self.y1_light = FIRST_BRANCH_HEIGHT + self.light_width
            else:
                self.y1_light -= LIGHT_DISPLACEMENT
                self.y2_light = self.y1_light - self.light_width

        elif np.around(self.x1_light, 2) >= 1 - self.light_width: # Indication that we are Right
            if np.around(self.y1_light,2) > 1 - LIGHT_DISPLACEMENT:
                print('check1')
                self.x1_light = 1-self.light_width
                print("x1_light",self.x1_light)
                self.x2_light = self.x1_light + self.light_width
            else:
                self.y1_light += LIGHT_DISPLACEMENT
                self.y2_light = self.y1_light - self.light_width

        else:
            self.x1_light -= LIGHT_DISPLACEMENT  # move by defined amount of pixels
            self.x2_light = self.x1_light + self.light_width



    #@staticmethod
    #@jit(nopython=True)
    #@partial(jit, static_argnums=(0,))
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

        elif self.y2_light + self.light_width >=1:
        #else:
            self.light_width += LIGHT_W_INCREMENT

    def tree_grow(self,x, y, mindist, maxdist):

        # apply filter to both idx and branches
        global branches_trimmed
        for i in range(len(x) - 1, 0, -1):  # number of possible scatters, check if they allow for branching with min_dist
            closest_branch = 0
            dist = 1
            #print('check')
            if len(self.branches) > MAX_BRANCHING:
                branches_trimmed = sample(self.branches, MAX_BRANCHING)
                #print('check2')
            else:
                branches_trimmed = self.branches


            branch_idx = [branch_idx for branch_idx, branch in enumerate(branches_trimmed)if self.x1_light <= branch.x2 <= self.x2_light or
                          self.y2_light <= branch.y2 <= self.y1_light]
            #print("this is branches trimmed:", branches_trimmed)
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
                #print('here', branches_trimmed)
                branches_trimmed[closest_branch].grow_count += 1
                branches_trimmed[closest_branch].grow_x += (
                    x[i] - branches_trimmed[closest_branch].x2) / (dist/BRANCH_LENGTH)
                branches_trimmed[closest_branch].grow_y += (
                    y[i] - branches_trimmed[closest_branch].y2) / (dist / BRANCH_LENGTH)
                #print('here', branches_trimmed)

        #print('again:', branches_trimmed)

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
        dist = distance.cdist(coords, [self.target], 'euclidean')

        # Get smallest distance to target
        min_dist = min(dist)
        #print(min_dist)

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
            #print(type(tree))
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
            #print('what is the dim',len(self.mnist_shape))
            #mnist = np.where(self.mnist_shape < 255, 100, img)
            #test = np.sum(mnist+img)

            #print(mnist)
            #print(mnist)

            yellow = (0, 128, 128)  # RGB color (dark yellow)
            x1 = ir(self.x1_light * self.width)
            x2 = ir(self.x2_light * self.width)

            y1 = ir(self.y1_light * self.height)
            y2 = ir(self.y2_light * self.height)
            if self.x2_light <= LIGHT_DISPLACEMENT or self.x1_light >= 1-self.light_width:
                cv2.rectangle(
                    img, pt1=(0, y1), pt2=(self.width, y2), color=yellow, thickness=-1)
            #elif self.
            else:
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
                #print("branch width", branch.width, " BRANCH THICCNESS: ", BRANCH_THICCNESS, " width: ", self.width)
                cv2.line(
                    img,
                    pt1=pt1,
                    pt2=pt2,
                    color=(0, 255, 0),
                    thickness=thiccness)
                # print(f"drawing branch from {pt1} to {pt2} "
                #       f"with thiccness {branch.width/50 * self.width}")

            z = np.where(self.mnist_shape < 255, img, 150)
            #img = np.sum(mnist, img)
            #for i in range(len(img)):
                #if self.mnist_shape[i] == 255:
                    #img[i] = self.mnist_shape[i]
                #img[i] = self.mnist_shape[i]
            # place goal as filled circle with center and radiusw
            # also important - place goal last because must be always visible
            #x = ir(self.target[0] * self.width)
            #y = ir(self.target[1] * self.height)
            # no target in this level
            #cv2.circle(
               # img,
               # center=(x, y),
               # radius=ir(.03 * self.width),
               # color=(0, 0, 255),
               # thickness=-1)
            # print(f"drawing goal circle at {(x, y)} "
            #       f"with radius {ir(.03*self.width)}")

            # flip image, because plant grows from the bottom, not the top

            #print(mnist)
            #print(img)
            #img = img+mnist
            #print('this is img',img[0])
            img = cv2.flip(z, 0)
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

        if random_start > .87:
            self.x1_light = .75
        elif random_start < 0.13:
            self.x1_light = 0
        else:
            self.x1_light = random_start - (LIGHT_WIDTH/2)

        if self.level == 'third' or self.level == 'second':
            print('here')
            start_light = np.random.rand()
            if start_light > .87:
                self.x1_light = .75
            elif start_light < 0.13:
                self.x1_light = 0
            else:
                self.x1_light = start_light - (LIGHT_WIDTH / 2)

        self.x2_light = self.x1_light + LIGHT_WIDTH
        self.light_width = LIGHT_WIDTH

        self.x_scatter = np.random.uniform(0, 1, LIGHT_DIF)
        self.y_scatter = np.random.uniform(0.25, 1, LIGHT_DIF)
        self.steps = 0
        self.new_branches = 0
        self.tips_per_step = 0
        self.y1_light = 1 #d+LIGHT_DISPLACEMENT
        self.y2_light = self.y1_light - LIGHT_WIDTH
        self.tips = [self.branches[0].x2, self.branches[0].y2]
        print("this is light displacement", LIGHT_DISPLACEMENT)


        # print("test",target_img)
        self.mnist_pixels = (self.get_observation()[:,:,2]/150)  # binary map of mnist shape
        self.plant_original = (self.get_observation()[:,:,1]/255) # binary map of original stem
        #print('this is mnist_img',mnist_pixels)
        #self.mnist_pixels = np.where(self.mnist_shape < 255,
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

        #if self.x2_light <= LIGHT_WIDTH or self.x1_light >= 1-LIGHT_WIDTH:
            #self.y2_light = self.y1_light - LIGHT_WIDTH
        #else:
            #self.x2_light = self.x1_light + LIGHT_WIDTH

        if action == 4:
            # then we keep the light in place
            pass


        if self.x2_light <= LIGHT_DISPLACEMENT or self.x1_light >= 1-LIGHT_WIDTH:  # light is in horizontal position
            try:
                convex_tips = np.array(self.tips)
                xs, ys = self.light_scatter()
                if len(convex_tips) >= 2:
                    hull = ConvexHull(convex_tips)
                    xxs = convex_tips[hull.vertices,0]    # x coords for convex hull around tips
                    yys = convex_tips[hull.vertices,1]    # y coords for convex hull around tips
                    y_max_idx = np.where(max(yys))        # idx where highest tip is
                    x_ymax = xxs[y_max_idx]

                    if self.y2_light < yys[y_max_idx]:   # if within the horizontal beam


                        if self.x2_light <= LIGHT_DISPLACEMENT:
                            filter = np.logical_and(xs >= 0, xs <= x_ymax)
                            xs = xs[filter]
                            ys = ys[filter]

                        if self.x1_light >= 1-LIGHT_WIDTH:
                            filter = np.logical_and(xs >= x_ymax, xs <= 1)
                            xs = xs[filter]
                            ys = ys[filter]
            except:
                pass
        else:
            xs, ys = self.light_scatter()


        # Branching step for light in this position
        tips = self.tree_grow(xs, ys, .01, .15)

        # Render image of environment at current state
        observation = self.get_observation()  #image
        plant = (observation[:,:,1]/255) # binary map of plant
        plant[plant>0.6] =1
        plant = plant.astype(int)
        print('plant', type(plant))
        true_plant = np.subtract(plant,self.plant_original)
        mnist = (observation[:,:,2]/150) # binary map of mnist
        mnist[mnist>0.5] =1
        mnist = mnist.astype(int)
        print('mnist',np.ndim(mnist))
        #intersection = np.all(true_plant == mnist) ### need to fix intersection

        check = np.sum((true_plant, mnist), axis=0)
        intersection = np.sum(np.where(check < 2, 0, 1))
        print('intersection', intersection)
        union = np.sum(np.where(check<2,check,1))
        print('union',union)
        reward = intersection / union

        #print(union)
        #self.mnist_img
        #reward = np.where(observation)
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
        print('y1:', self.y1_light)
        print('y2:',self.y2_light)
        print('x1:', self.x1_light)
        print('x2:',self.x2_light)
        #print(misc)
        #self.number_of_branches = new_branches
        #print("how many new branches? ", misc['new_branches'])
        return observation, reward, done, misc

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

    #gse = GrowSpaceEnv(obs_type = None, level= 'third')
    gse = gym.make('GrowSpaceEnv-Mnist-v0')
    #gse = GrowSpaceEnv()

    def key2action(key):
        if key == ord('a'):
            return 0  # move left
        elif key == ord('d'):
            return 1  # move right
        elif key == ord('s'):
            return 4  # stay in place
        elif key == ord('w'):
            return 2
        elif key == ord('x'):
            return 3
        else:
            return None

    rewards = []
    print('what is this')
    while True:
        gse.reset()
        img = gse.get_observation(debug_show_scatter=False)
        #image = img.astype(np.uint8)
        #backtorgb = image * 255
        #print(backtorgb)
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

        print("amount of rewards:", total)        # cv2.waitKey(1)  # this is necessary or the window closes immediately
        #else:
            #dreturn img








