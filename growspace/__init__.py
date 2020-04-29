from gym import register

register(
    id=f'GrowSpaceEnv-States-v0',
    entry_point='growspace.envs:GrowSpaceEnv',
    kwargs={
        "observe_images": False
    },
    max_episode_steps=100, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Image-v0',
    entry_point='growspace.envs:GrowSpaceEnv',
    kwargs={
        "observe_images": True
    },
    max_episode_steps=100, # number of total steps in the environment
)