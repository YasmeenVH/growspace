import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("GrowSpaceEnv-Images-v0")


def run_episode():
    timers_step = []
    for step in range(25):        # defined in __init__.py    # here we're timing the step function
        start = time.time()
        env.step(env.action_space.sample())
        diff = time.time() - start
        timers_step.append(diff)
        #print(timers_step)
    #print(f"step times: {np.mean(timers_step)} +/- {np.std(timers_step)}")
    return timers_step

def plot_episode(timings):
    plt.plot(timings)
    plt.xlabel("steps")
    plt.ylabel("time (s)")
    plt.show()

def run_many_episodes(episodes):
    timers_reset = []
    timings = []
    for _ in range(episodes):  # here we're timing the reset function
        start = time.time()
        env.reset()
        diff = time.time() - start
        timers_reset.append(diff)
        t = run_episode()
        timings.append(t)

    print(f"reset times: {np.mean(timers_reset)} +/- {np.std(timers_reset)}")
    print(f"step times: {np.mean(timings)} +/- {np.std(timings)}")
    return timings

def plot(timings):
    av = np.mean(timings,axis=0)
    err = np.std(timings,axis=0)
    x = np.arange(0,25)
    print(len(x))
    print(len(av))
    fig, ax = plt.subplots(1)
    ax.plot(x, av)
    ax.fill_between(x, av+err, av-err, alpha=0.5)
    ax.set_title(r'over 10 episodes')
    ax.set_xlabel("steps")
    ax.set_ylabel("seconds")
    ax.grid()
    plt.show()


if __name__ == '__main__':
  t = run_many_episodes(10)
  plot(t)
