import enum

import cv2
import numpy as np
from growspace.envs.growspace_control import GrowSpaceEnv_Control

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


class Actions(enum.IntEnum):
    move_left = 0
    move_right = 1
    increase_beam = 2
    decrease_beam = 3
    noop = 4
    water = 5


class GrowSpaceWater(GrowSpaceEnv_Control):
    def __init__(self):
        self.max_water_level = 1
        self.water_level = self.max_water_level
        self.branch_water_usage = 0.1
        self.paint_water = False
        super().__init__()

    def step(self, action):
        resource_cost = 0.
        self.paint_water = False
        if action == Actions.water:
            self.water_level = self.max_water_level
            action = Actions.noop
            resource_cost += 1.
            self.paint_water = True

        s, r, t, i = super().step(action)

        r -= resource_cost
        if i['new_branches']:
            self.water_level -= i['new_branches'] * self.branch_water_usage
        return s, r, t, i

    def get_observation(self, *args, **kwargs):
        obs = super().get_observation(*args, **kwargs)
        if self.paint_water:
            obs[-5:-1, -5:-1, 0] = 255
        return obs

    def tree_grow(self, *args, **kwargs):
        if self.water_level <= 0:
            return [[branch.x2, branch.y2] for branch in self.branches]

        return super().tree_grow(*args, **kwargs)

    def reset(self):
        self.water_level = self.max_water_level
        super().reset()

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

    env = GrowSpaceWater()


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
        elif key == ord('e'):
            return 5
        else:
            return None


    rewards = []
    while True:
        env.reset()
        img = env.get_observation(debug_show_scatter=False)
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

            b, t, c, f = env.step(action)
            print(f["new_branches"])
            rewards.append(t)
            cv2.imshow("plant", env.get_observation(debug_show_scatter=False))
        total = sum(rewards)
        print("amount of rewards:", total)
