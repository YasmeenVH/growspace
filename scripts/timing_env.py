import gym
import time
import growspace
import numpy as np
import matplotlib.pyplot as plt
from scripts.control_old import GrowSpaceEnv_Control

env1 = gym.make("GrowSpaceEnv-Control-v0")
env2 = GrowSpaceEnv_Control()


def run_episode(env):
    timers_step = []
    for step in range(50):        # defined in __init__.py    # here we're timing the step function
        #print(step)
        start = time.time()
        env.step(env.action_space.sample())
        diff = time.time() - start
        timers_step.append(diff)

    print(f"step times: {np.mean(timers_step)} +/- {np.std(timers_step)}")
    return timers_step

def plot_episode(timings):
    plt.plot(timings)
    plt.xlabel("steps")
    plt.ylabel("time (s)")
    plt.show()

def run_many_episodes(episodes,env):
    timers_reset = []
    timings = []
    for _ in range(episodes):  # here we're timing the reset function
        start = time.time()
        env.reset()
        diff = time.time() - start
        timers_reset.append(diff)
        t = run_episode(env)
        timings.append(t)
        #print("what is timin)
    print(f"reset times: {np.mean(timers_reset)} +/- {np.std(timers_reset)}")
    print(f"step times: {np.mean(timings)} +/- {np.std(timings)}")

    return timings

def plot(timings):
    av = np.mean(timings,axis=0)
    err = np.std(timings,axis=0)
    x = np.arange(0,50)


    fig, ax = plt.subplots(1)
    ax.plot(x, av)
    ax.fill_between(x, av+err, av-err, alpha=0.5)
    ax.set_title(r'over 100 episodes, with capping of 20 branches')
    ax.set_ylabel("seconds")
    ax.grid()
    plt.show()


if __name__ == '__main__':
  t = run_many_episodes(100,env1)
  av = np.mean(t, axis=0)
  #t2 = run_many_episodes(100)
  t2 = run_many_episodes(100,env2)
  av2 = np.mean(t2, axis=0)
  fig, ax = plt.subplots()
  ax.set_title('Timing over 100 episodes')
  data = [av,av2]
  ax.boxplot(data)

  plt.show()

