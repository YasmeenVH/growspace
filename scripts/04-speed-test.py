import time

from tqdm import trange

import growspace
import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("GrowSpaceEnv-Images-v0")

timings = {"steps": [], "rendering": [], "resets": []}

for i in trange(10):
    start = time.time()
    env.reset()
    diff = time.time() - start
    timings["resets"].append(diff)
    episode_steps = []
    episode_rend = []
    for _ in range(25):
        action = env.action_space.sample()

        start = time.time()
        env.step(action)
        diff = time.time() - start
        episode_steps.append(diff)

        start = time.time()
        env.render("rgb_array")
        diff = time.time() - start
        episode_rend.append(diff)
    timings["steps"].append(episode_steps)
    timings["rendering"].append(episode_rend)

steps = np.array(timings["steps"])
rend = np.array(timings["rendering"])

steps = [d for d in steps.T[:]]
rend = [d for d in rend.T[:]]

steps_wo_rend = []
for i in range(25):
    steps_wo_rend.append(steps[i] - rend[i])
#
# # Multiple box plots on one Axes
# fig, ax = plt.subplots(2, 2)
# ax[0][0].boxplot(timings["resets"])
# ax[0][0].set_title("Resets")
# ax[0][0].set_ylabel("seconds")
#
# ax[0][1].boxplot(steps)
# ax[0][1].set_title("Steps")
# ax[0][1].set_ylabel("seconds")
# ax[0][1].set_xlabel("episode step")
#
#
# ax[1][0].boxplot(rend)
# ax[1][0].set_title("Rendering")
# ax[1][0].set_ylabel("seconds")
# ax[1][0].set_xlabel("episode step")
#
# ax[1][1].boxplot(steps_wo_rend)
# ax[1][1].set_title("Steps w/o Rendering")
# ax[1][1].set_ylabel("seconds")
# ax[1][1].set_xlabel("episode step")
#
#
# plt.tight_layout()
# plt.show()
#
# # --------------------- Hz
#
# fig, ax = plt.subplots(2, 2)
# ax[0][0].boxplot(1/np.array(timings["resets"]))
# ax[0][0].set_title("Resets")
# ax[0][0].set_ylabel("Hz")
#
# steps = [1/s for s in steps]
# ax[0][1].boxplot(steps)
# ax[0][1].set_title("Steps")
# ax[0][1].set_ylabel("Hz")
# ax[0][1].set_xlabel("episode step")
# ax[0][1].set_ylim(0,2000)
#
# rend = [1/r for r in rend]
# ax[1][0].boxplot(rend)
# ax[1][0].set_title("Rendering")
# ax[1][0].set_ylabel("Hz")
# ax[1][0].set_xlabel("episode step")
# ax[1][0].set_ylim(0,2000)
#
# swor = [1/s for s in steps_wo_rend]
# ax[1][1].boxplot(swor)
# ax[1][1].set_title("Steps w/o Rendering")
# ax[1][1].set_ylabel("Hz")
# ax[1][1].set_xlabel("episode step")
# ax[1][1].set_ylim(0,2000)
#
# for c in ax:
#    for a in c:
#       a.axhline(y=1000, color='y')
#       a.axhline(y=100, color='r')
#
# plt.tight_layout()
# plt.show()
