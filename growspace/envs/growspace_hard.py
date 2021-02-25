from random import sample

import cv2
import gym
import numpy as np
from growspace.plants.tree import Branch
from numpy.linalg import norm
from scipy.spatial import distance

FIRST_BRANCH_HEIGHT = .24
BRANCH_THICCNESS = .015
BRANCH_LENGTH = 1 / 9
MAX_BRANCHING = 10
DEFAULT_RES = 84
LIGHT_WIDTH = .25
LIGHT_DIF = 250
LIGHT_DISPLACEMENT = .1
LIGHT_W_INCREMENT = .1
MIN_LIGHT_WIDTH = .1
MAX_LIGHT_WIDTH = .5
MAX_STEPS = 25


def to_int(v):
    return int(round(v))


class Actions(Enum):
    move_left = 0
    move_right = 1
    increase_beam = 2
    decrease_beam = 3
    noop = 4


class GrowSpaceEnv_Lia(GrowSpaceEnv_Control):
    def __init__(self):
        self.width = DEFAULT_RES
        self.height = DEFAULT_RES
        self.light_dif = LIGHT_DIF
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.obs_type = obs_type
        self.observation_space = gym.spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)

    def tree_grow(self, x, y, mindist, maxdist):
        global branches_trimmed
        for i in range(len(x) - 1, 0, -1):  # number of possible scatters, check if they allow for branching with min_dist
            closest_branch = 0
            dist = 1

            if len(self.branches) > MAX_BRANCHING:
                branches_trimmed = sample(self.branches, MAX_BRANCHING)
            else:
                branches_trimmed = self.branches
            branch_idx = [branch_idx for branch_idx, branch in enumerate(branches_trimmed) if self.x1_light <= branch.x2 <= self.x2_light]
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
                                                                   x[i] - branches_trimmed[closest_branch].x2) / (dist / BRANCH_LENGTH)
                branches_trimmed[closest_branch].grow_y += (
                                                                   y[i] - branches_trimmed[closest_branch].y2) / (dist / BRANCH_LENGTH)

        # print('branches trimmed', branches_trimmed)
        # if branches_trimmed == 0:
        # pass
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
        # sending coordinates out
        for branch in self.branches:
            # x2 and y2 since they are the tips
            branch_coords.append([branch.x2, branch.y2])
        # print("branching has occured")

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

            yellow = (0, 128, 128)  # RGB color (dark yellow)
            x1 = to_int(self.x1_light * self.width)
            x2 = to_int(self.x2_light * self.width)
            cv2.rectangle(
                img, pt1=(x1, 0), pt2=(x2, self.height), color=yellow, thickness=-1)
            light_img = np.sum(img, axis=2)
            light = np.where(light_img <= 128, light_img, 1)

            # ---tree--- #
            img1 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for branch in self.branches:
                pt1, pt2 = branch.get_pt1_pt2()
                thiccness = to_int(branch.width * BRANCH_THICCNESS * self.width)
                cv2.line(
                    img1,
                    pt1=pt1,
                    pt2=pt2,
                    color=(0, 255, 0),
                    thickness=thiccness)
            tree_img = np.sum(img1, axis=2)
            tree = np.where(tree_img < 255, tree_img, 1)

            # ---light + tree ----#
            light_tree = light + tree  # addition of matrices
            light_tree_binary = np.where(light_tree < 2, light_tree, 1)
            # ---target--- #
            img2 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            x = to_int(self.target[0] * self.width)
            y = to_int(self.target[1] * self.height)
            cv2.circle(
                img2,
                center=(x, y),
                radius=to_int(.03 * self.width),
                color=(0, 0, 255),
                thickness=-1)

            target_img = np.sum(img2, axis=2)
            target = np.where(target_img < 255, target_img, 1)
            light_target = light + target  # additions of matrices
            light_target_binary = np.where(light_target < 2, light_target, 1)
            final_img = np.dstack((light, light_tree_binary, light_target_binary))
            final_img = cv2.flip(final_img, 0)
            # print("dimensions of final shape", np.shape(final_img))
            return final_img

        if self.obs_type == None:
            # place light as rectangle
            yellow = (0, 128, 128)  # RGB color (dark yellow)
            x1 = to_int(self.x1_light * self.width)
            x2 = to_int(self.x2_light * self.width)
            cv2.rectangle(
                img, pt1=(x1, 0), pt2=(x2, self.height), color=yellow, thickness=-1)

            if debug_show_scatter:
                xs, ys = self.light_scatter()
                for k in range(len(xs)):
                    x = to_int(xs[k] * self.width)
                    y = to_int(ys[k] * self.height)
                    cv2.circle(
                        img,
                        center=(x, y),
                        radius=2,
                        color=(255, 0, 0),
                        thickness=-1)

            # Draw plant as series of lines (1 branch = 1 line)
            for branch in self.branches:
                pt1, pt2 = branch.get_pt1_pt2()
                thiccness = to_int(branch.width * BRANCH_THICCNESS * self.width)
                cv2.line(
                    img,
                    pt1=pt1,
                    pt2=pt2,
                    color=(0, 255, 0),
                    thickness=thiccness)

            x = to_int(self.target[0] * self.width)
            y = to_int(self.target[1] * self.height)
            cv2.circle(
                img,
                center=(x, y),
                radius=to_int(.03 * self.width),
                color=(0, 0, 255),
                thickness=-1)

            img = cv2.flip(img, 0)

            # print("dimensions of final shape", np.shape(img))
            return img

    def reset(self):
        # Set env back to start - also necessary on first start
        # is in range [0,1]
        if self.setting == 'easy':
            random_start = 0.07
            self.target = [random_start, .8]

        elif self.setting == 'hard':
            random_start = 0.07
            self.target = [1 - random_start, .8]
        else:
            random_start = np.random.rand()  # is in range [0,1]
            self.target = [random_start, .8]
        self.branches = [
            Branch(
                x=random_start,
                x2=random_start,
                y=0,
                y2=FIRST_BRANCH_HEIGHT,
                img_width=self.width,
                img_height=self.height)]

        # self.target = [np.random.uniform(0, 1), np.random.uniform(.8, 1)]
        # self.target = [np.random.uniform(0, 1), .8]
        self.light_width = .25
        if random_start > .87:
            self.x1_light = .75
        elif random_start < 0.13:
            self.x1_light = 0
        else:
            self.x1_light = random_start - (self.light_width / 2)

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
        self.y_scatter = np.random.uniform(0.25, 1, self.light_dif)
        self.steps = 0
        self.new_branches = 0
        self.tips_per_step = 0
        self.light_move = 0

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
        # print("light x1", self.x1_light)
        # print('action',action)
        self.x2_light = self.x1_light + self.light_width

        # filter scattering
        xs, ys = self.light_scatter()
        # print("scattering x len:", len(xs))
        # print("this is lightx1 :", self.x1_light, "and light width:", self.light_width)
        # Branching step for light in this position
        tips = self.tree_grow(xs, ys, .01, .15)
        # print("tips:", tips)
        # Calculate distance to target
        if self.distance_target(tips) <= 0.1:
            reward = 1 / 0.1 / 10
            success = 1
            # reward = preprocessing.normalize(reward)
        else:
            reward = 1 / self.distance_target(tips) / 10
            success = 0
        # print("this is reward:",reward)
        # reward = preprocessing.normalize(reward)
        # Render image of environment at current state
        observation = self.get_observation()  # image
        # print("these are tips:",tips)
        # print("length of tips:", len(tips))

        done = self.steps == MAX_STEPS
        misc = {"tips": tips, "target": self.target, "light": self.x1_light, "light_width": self.light_width, "step": self.steps, "success": success}

        if self.steps == 0:
            self.new_branches = len(tips)
            misc['new_branches'] = self.new_branches
            self.light_move = self.light_move

        else:
            new_branches = len(tips) - self.new_branches
            misc['new_branches'] = new_branches
            self.new_branches = len(tips)  # reset for future step
            self.light_move = np.abs(self.light_move - self.x1_light)
            misc['light_move'] = self.light_move

        misc['img'] = observation
        # print(self.light_width, "this is light width")
        # (optional) additional information about plant/episode/other stuff, leave empty for now
        # print("steps:", self.steps)    # sanity check
        self.steps += 1
        # print(misc)
        # self.number_of_branches = new_branches
        # print("how many new branches? ", misc['new_branches'])
        # print("what type of data", type(misc['new_branches']))
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

    gse = GrowSpaceEnv_Control()


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
    while True:
        gse.reset()
        img = gse.get_observation(debug_show_scatter=False)
        image = img.astype(np.uint8)
        backtorgb = image * 255
        print(backtorgb)
        cv2.imshow("plant", img)
        rewards = []
        c = False
        while not c:
            action = key2action(cv2.waitKey(-1))
            if action is None:
                quit()

            b, t, c, f = gse.step(action)
            print(f["new_branches"])
            rewards.append(t)
            cv2.imshow("plant", gse.get_observation(debug_show_scatter=False))
        total = sum(rewards)
        print("amount of rewards:", total)
