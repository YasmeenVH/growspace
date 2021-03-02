import enum

import cv2
import gym
import numpy as np
from growspace.envs.growspace_control import GrowSpaceEnv_Control
from growspace.envs.growspace_resources import Actions

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


class ContinuousActions(enum.IntEnum):
    light_velocity = 0
    beam_width = 1
    # water = 2 # TODO add water -> 3 actions


class GrowSpaceContinuous(GrowSpaceEnv_Control):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(-1, 1, (len(ContinuousActions),))

    def step(self, action: np.ndarray):
        desired_light_displacement = action[ContinuousActions.light_velocity]
        beam_width_change = action[ContinuousActions.beam_width]
        self.continous_light_move(desired_light_displacement)
        self.continous_light_width_change(beam_width_change)
        step, reward, terminal, info = super().step(Actions.noop)
        return step, reward, terminal, info

    def continous_light_move(self, desired_light_displacement):
        new_x1_light = self.x1_light + desired_light_displacement
        new_x2_light = new_x1_light + self.light_width

        if new_x2_light > 1:
            right_overflow = (new_x2_light - 1)
            new_x1_light -= right_overflow

        if new_x1_light < 0:
            left_overflow = -new_x1_light
            new_x1_light = 0

        self.x1_light = new_x1_light

    def continous_light_width_change(self, beam_change):
        new_width = self.light_width + beam_change
        new_width_clipped = np.clip(new_width, MIN_LIGHT_WIDTH, MAX_LIGHT_WIDTH)
        right_overflow = max(0, self.x1_light + new_width_clipped - 1)
        self.light_width = new_width_clipped - right_overflow
        assert self.light_width + self.x1_light <= 1 and self.x1_light >= 0, "Ouch! light bean is wide, it's outside the sides"


def enjoy():
    env = GrowSpaceContinuous()

    def key2continuous_action(key):
        if key == ord('a'):
            return np.array([-0.25, 0])
        elif key == ord('d'):
            return np.array([0.25, 0])
        elif key == ord('q'):
            return np.array([-0.25, -0.1])
        elif key == ord('e'):
            return np.array([0.25, 0.1])
        elif key == ord('z'):
            return np.array([0, -0.1])
        elif key == ord('c'):
            return np.array([0, 0.1])
        else:
            return None

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
            action = key2continuous_action(cv2.waitKey(-1))
            if action is None:
                quit()

            b, t, c, f = env.step(action)
            print(f["new_branches"])
            rewards.append(t)
            cv2.imshow("plant", env.get_observation(debug_show_scatter=False))
        total = sum(rewards)
        print("amount of rewards:", total)


if __name__ == '__main__':
    enjoy()
