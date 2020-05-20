import cv2
import gym
import numpy as np
from numpy.linalg import norm

from growspace.plants.tree import Branch
from scipy.spatial import distance

FIRST_BRANCH_HEIGHT = .24
BRANCH_THICCNESS = .013
BRANCH_LENGTH = 1/9

def to_int(v):
    return int(round(v))


ir = to_int  # shortcut for function call


class GrowSpaceEnv(gym.Env):

    def __init__(self, width=84, height=84, light_dif=250):
        self.width = width  # do we keep?
        self.height = height  # do we keep?
        self.seed()
        self.light_dif = light_dif
        self.action_space = gym.spaces.Discrete(3)  # L, R, keep of light paddle
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self.steps = 0

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

    def tree_grow(self, x, y, mindist, maxdist):
        # available scatter due to light position (look through xcoordinates
        for i in range(len(x) - 1, 0, -1):
            closest_branch = 0
            dist = 1

            # loop through branches and see which coordinate within available scatter is the closest
            for branch_idx, branch in enumerate(self.branches):
                temp_dist = norm([x[i] - branch.x2,
                                  y[i] - branch.y2])  #euclidean distance
                if temp_dist < dist:
                    dist = temp_dist
                    closest_branch = branch_idx

            # removes scatter points if reached
            if dist < mindist:
                x = np.delete(x, i)
                y = np.delete(y, i)

            # when distance is greater than max distance, branching occurs to find other points.
            elif dist < maxdist:

                self.branches[closest_branch].grow_count += 1
                self.branches[closest_branch].grow_x += (
                    x[i] - self.branches[closest_branch].x2) / (dist/BRANCH_LENGTH)
                self.branches[closest_branch].grow_y += (
                    y[i] - self.branches[closest_branch].y2) / (dist/BRANCH_LENGTH)
                # print(f"closest branch: {closest_branch}\n"
                #       f"grow_count = {self.branches[closest_branch].grow_count}\n"
                #       f"grow_x* = {(x[i] - self.branches[closest_branch].x2) * dist}\n"
                #       f"grow_x/ = {(x[i] - self.branches[closest_branch].x2) / (dist*10)}")

        # generation of new branches (forking) in previous step will generate a new branch with grow count
        for i in range(len(self.branches)):
            if self.branches[i].grow_count > 0:
                newBranch = Branch(
                    self.branches[i].x2, self.branches[i].x2 +
                    self.branches[i].grow_x / self.branches[i].grow_count,
                    self.branches[i].y2, self.branches[i].y2 +
                    self.branches[i].grow_y / self.branches[i].grow_count,
                    self.width, self.height)
                self.branches.append(newBranch)
                self.branches[i].child.append(newBranch)
                self.branches[i].grow_count = 0
                self.branches[i].grow_x = 0
                self.branches[i].grow_y = 0

        # increase thickness of first elements added to tree as they grow
        self.branches[0].update_width()
        branch_coords = []

        #sending coordinates out
        for branch in self.branches:
            # x2 and y2 since they are the tips
            branch_coords.append([branch.x2, branch.y2])

        return branch_coords

    def distance_target(self, coords):
        # Calculate distance from each tip grown
        dist = distance.cdist(coords, [self.target],
                              'euclidean')  #TODO replace with numpy

        # Get smallest distance to target
        min_dist = min(dist)

        return min_dist

    def get_observation(self, debug_show_scatter=False):
        # new empty image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # place light as rectangle
        yellow = (0, 128, 128)  # RGB color (dark yellow)
        x1 = ir(self.x1_light * self.width)
        x2 = ir(self.x2_light * self.width)
        cv2.rectangle(
            img, pt1=(x1, 0), pt2=(x2, self.height), color=yellow, thickness=-1)
        # print(f"drawing light rectangle from {(x1, 0)} "
        #       f"to {(x2, self.height)}")

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
            print("branch width", branch.width, " BRANCH THICCNESS: ", BRANCH_THICCNESS, " width: ", self.width)
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

        return img

    def reset(self):
        # Set env back to start - also necessary on first start
        random_start = np.random.rand()  # is in range [0,1

        self.branches = [
            Branch(
                x=random_start,
                x2=random_start,
                y=0,
                y2=FIRST_BRANCH_HEIGHT,
                img_width=self.width,
                img_height=self.height)
        ]
        self.target = np.array(
            [np.random.uniform(0, 1),
             np.random.uniform(.8, 1)])
        self.light_width = .25
        self.x1_light = .4
        self.x2_light = self.x1_light + self.light_width

        self.x_scatter = np.random.uniform(0, 1, self.light_dif)
        self.y_scatter = np.random.uniform(0.25, 1, self.light_dif)
        self.steps = 0

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

        # Calculate distance to target
        reward = 1 / self.distance_target(tips)

        # Render image of environment at current state
        observation = self.get_observation()  #image

        done = False  # because we don't have a terminal condition
        misc = {
        }  # (optional) additional information about plant/episode/other stuff, leave empty for now
        print("steps:", self.steps)
        self.steps += 1
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

    while True:
        gse.reset()
        img = gse.get_observation(debug_show_scatter=True)
        cv2.imshow("plant", img)

        for _ in range(20):
            action = key2action(cv2.waitKey(-1))
            if action is None:
                quit()

            gse.step(action)
            cv2.imshow("plant", gse.get_observation(debug_show_scatter=True))
