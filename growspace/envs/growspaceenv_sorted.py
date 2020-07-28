from random import sample

import cv2
import gym
import numpy as np
from numpy.linalg import norm
from sortedcontainers import SortedDict

from growspace.defaults import (
    DEFAULT_RES,
    LIGHT_DIFFUSION,
    FIRST_BRANCH_HEIGHT,
    LIGHT_WIDTH,
    EPSILON,
    BRANCH_THICCNESS,
    POINT_RADIUS,
    LIGHT_COLOR,
    POINT_COLOR,
    PLANT_COLOR,
    LIGHT_STEP,
    MAX_GROW_DIST,
    MIN_GROW_DIST,
    BRANCH_LENGTH,
    MAX_BRANCHES,
)
from growspace.utils import ir


class GrowSpaceSortedEnv(gym.Env):
    def __init__(self, width=DEFAULT_RES, height=DEFAULT_RES, light_dif=LIGHT_DIFFUSION):
        self.width = width
        self.height = height
        self.seed()
        self.light_dif = light_dif
        self.action_space = gym.spaces.Discrete(3)  # L, R, keep of light paddle
        self.observation_space = gym.spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)
        self.steps = 0

        # data format for branches: they are indexed/sorted by x_end position and each
        # key has a list of values that are [y_end, x_start, y_start, children]

        self.branches = SortedDict()
        self.points = SortedDict()

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def light_move_R(self):
        if np.around(self.light_left, 1) >= 1 - LIGHT_WIDTH - LIGHT_STEP:  # limit of coordinates
            self.light_left = 1 - LIGHT_WIDTH  # stay put
        else:
            self.light_left += LIGHT_STEP  # move by .1 right

    def light_move_L(self):
        if np.around(self.light_left, 1) <= LIGHT_STEP:  # limit of coordinates
            self.light_left = 0
        else:
            self.light_left -= LIGHT_STEP  # move by .1 left

    def find_closest_branch(self, point_x, branches):
        branch_names = []
        branch_distances = []
        # prefilter by x
        if len(branches) > MAX_BRANCHES:
            branches_trimmed = sample(branches, MAX_BRANCHES)
        else:
            branches_trimmed = branches
        for branch in branches_trimmed:
            dist_x = branch - point_x
            if np.abs(dist_x) <= MAX_GROW_DIST:
                # we got a potential candidate - now let's check Y
                dist_y = self.branches[branch][0] - self.points[point_x]
                if np.abs(dist_y) <= MAX_GROW_DIST:
                    dist = norm((dist_x, dist_y))
                    if dist <= MAX_GROW_DIST:
                        branch_names.append(branch)
                        branch_distances.append(dist)
        if len(branch_distances) == 0:
            return None, None
        argmin = np.argmin(branch_distances)
        return branch_names[argmin], branch_distances[argmin]

    def grow_plant(self):
        points_filtered = list(
            self.get_points_in_range(self.light_left - MAX_GROW_DIST, self.light_right + MAX_GROW_DIST)
        )
        branches_filtered = list(self.get_branches_in_range(self.light_left, self.light_right))

        growths = {}  # will have the format: [(branch, target_x)]

        for point in points_filtered:
            closest_branch, dist = self.find_closest_branch(point, branches_filtered)
            if closest_branch is None:
                continue
            if dist < MIN_GROW_DIST:
                self.points.pop(point)
            elif dist < MAX_GROW_DIST:
                if closest_branch not in growths:
                    growths[closest_branch] = [point]
                else:
                    growths[closest_branch].append(point)

        for branch, points in growths.items():
            end_x = (
                branch + (sum(points) / len(points) - branch) * BRANCH_LENGTH
            )  # alternatively sum(poins)/len(points)
            branch_y = self.branches[branch][0]
            point_ys = [self.points[p] for p in points]
            end_y = branch_y + (sum(point_ys) / len(point_ys) - branch_y) * BRANCH_LENGTH
            while end_x in self.branches:
                end_x += EPSILON  # keys need to be unique in branches dict
            self.branches[end_x] = [end_y, branch, self.branches[branch][0], 0]

        # update_all_branch_widths(branches)

    def get_points_in_range(self, start, end):
        return self.points.irange(start, end)  # this is dark SortedDict magic

    def get_branches_in_range(self, start, end):
        return self.branches.irange(start, end)  # this is dark SortedDict magic

    def branch_bisect_range(self, lower, upper):
        start = self.branches.bisect(lower)
        end = self.branches.bisect_right(upper)
        return self.branches[start:end]

    def get_branch_start_end_thiccness(self, end_x):
        end_y, start_x, start_y, children = self.branches[end_x]
        thicc = ir((children + 1) * BRANCH_THICCNESS * self.width)
        return (
            (ir(start_x * self.width), ir(start_y * self.height)),
            (ir(end_x * self.width), ir(end_y * self.height)),
            thicc,
        )

    def get_observation(self, debug_show_scatter=False):
        # new empty image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # place light as rectangle
        x1 = ir(self.light_left * self.width)
        x2 = ir(self.light_right * self.width)
        cv2.rectangle(img, pt1=(x1, 0), pt2=(x2, self.height), color=LIGHT_COLOR, thickness=-1)

        if debug_show_scatter:
            points_filtered = self.get_points_in_range(self.light_left, self.light_right)
            for k in list(points_filtered):
                x = ir(k * self.width)
                y = ir(self.points[k] * self.height)
                cv2.circle(img, center=(x, y), radius=POINT_RADIUS, color=POINT_COLOR, thickness=-1)

        # Draw plant as series of lines (1 branch = 1 line)
        for branch_x_end in self.branches.keys():
            start, end, thiccness = self.get_branch_start_end_thiccness(branch_x_end)
            cv2.line(img, pt1=start, pt2=end, color=PLANT_COLOR, thickness=thiccness)

        # place goal as filled circle with center and radius
        # also important - place goal last because must be always visible
        x = ir(self.target[0] * self.width)
        y = ir(self.target[1] * self.height)
        cv2.circle(img, center=(x, y), radius=ir(0.03 * self.width), color=(0, 0, 255), thickness=-1)

        # flip image, because plant grows from the bottom, not the top
        img = cv2.flip(img, 0)

        return img

    def reset(self):
        random_start = np.random.rand()  # is in range [0,1
        self.branches.clear()
        self.points.clear()

        self.branches[random_start] = [FIRST_BRANCH_HEIGHT, random_start, 0, 0]

        self.target = [np.random.uniform(0, 1), np.random.uniform(0.8, 1)]
        if random_start >= (1 - LIGHT_WIDTH / 2):
            self.light_left = 1 - LIGHT_WIDTH
        elif random_start <= LIGHT_WIDTH / 2:
            self.light_left = 0
        else:
            self.light_left = random_start - (LIGHT_WIDTH / 2)

        self.light_right = self.light_left + LIGHT_WIDTH

        points_x = np.random.uniform(0, 1, self.light_dif)
        points_y = np.random.uniform(FIRST_BRANCH_HEIGHT + 0.1, 1, self.light_dif)

        for i in range(self.light_dif):
            while points_x[i] in self.points:
                points_x[i] += EPSILON
            self.points[points_x[i]] = points_y[i]

        self.steps = 0

        return self.get_observation()

    def step(self, action):
        # Two possible actions, move light left or right

        if action == 0:
            self.light_move_L()

        if action == 1:
            self.light_move_R()

        self.light_right = self.light_left + LIGHT_WIDTH

        if action == 2:
            # then we keep the light in place
            pass

        self.grow_plant()

        # # Calculate distance to target
        # reward = 1 / self.distance_target(tips)

        ####### TODO

        reward = 0  # TODO

        ####### TODO

        # Render image of environment at current state
        observation = self.get_observation()  # image

        done = False  # because we don't have a terminal condition
        misc = {}  # (optional) additional information about plant/episode/other stuff, leave empty for now
        # print("steps:", self.steps)    # sanity check
        self.steps += 1
        return observation, reward, done, misc

    def render(self, mode="human", debug_show_scatter=False):  # or mode="rgb_array"
        img = self.get_observation(debug_show_scatter)

        if mode == "human":
            cv2.imshow("plant", img)  # create opencv window to show plant
            cv2.waitKey(1)  # this is necessary or the window closes immediately
        else:
            return img


if __name__ == "__main__":
    import time

    gse = GrowSpaceSortedEnv()

    def key2action(key):
        if key == ord("a"):
            return 0  # move left
        elif key == ord("d"):
            return 1  # move right
        elif key == ord("s"):
            return 2  # stay in place
        else:
            return None

    while True:
        gse.reset()
        gse.render("human", True)

        for _ in range(20):
            action = key2action(cv2.waitKey(-1))
            if action is None:
                quit()

            gse.step(action)
            gse.render("human", True)
