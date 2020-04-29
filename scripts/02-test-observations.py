import gym  # need to create gym-based envs
import growspace  # to tell gym that it can also load your custom environment

env = gym.make("GrowSpaceEnv-Images-v0")

obs = env.reset()

print(obs.shape) # assuming what you return is a np.array() for pixels
print(obs.min(), obs.max()) # to see if it's normalized (between 0 and 1)
print(obs)

random_action = env.action_space.sample()

obs, rew, done, misc = env.step(random_action)

print(obs.shape) # assuming what you return is a np.array()
print(obs.min(), obs.max()) # to see if it's normalized
print(obs, rew)



