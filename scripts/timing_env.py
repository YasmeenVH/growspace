import gym
import time
import growspace
import numpy as np

env = gym.make("GrowSpaceEnv-Images-v0")
timers_reset = []
timers_step = []
for _ in range (3):         # here we're timing the reset function
  start = time.time()
  env.reset()
  diff = time.time() - start
  timers_reset.append(diff)
  for step in range(25):        # defined in __init__.py    # here we're timing the step function
    start = time.time()
    env.step(env.action_space.sample())
    diff = time.time() - start
    timers_step.append(diff)

print (f"reset times: {np.mean(timers_reset)} +/- {np.std(timers_reset)}")
print (f"step times: {np.mean(timers_step)} +/- {np.std(timers_step)}")

