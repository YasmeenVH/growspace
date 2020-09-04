from gym.envs.registration import register
#
# register(
#     id=f'GrowSpaceEnv-States-v0',
#     entry_point='growspace.envs:GrowSpaceEnv',
#     kwargs={
#         # "observe_images": False
#     },
#     max_episode_steps=200, # number of total steps in the environment
# )

register(
    id=f'GrowSpaceEnv-Images-v0',
    entry_point='growspace.envs:GrowSpaceEnv',
    kwargs={'obs_type': None,
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)
register(
    id=f'GrowSpaceEnv-Images-v1',
    entry_point='growspace.envs:GrowSpaceEnv',
    kwargs={'obs_type': 'Binary',
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)
register(
    id=f'GrowSpaceEnv-Images-v2',
    entry_point='growspace.envs:GrowSpaceEnv',
    kwargs={'obs_type': None, 'level': 'second',
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Images-v3',
    entry_point='growspace.envs:GrowSpaceEnv',
    kwargs={'obs_type': 'Binary', 'level': 'second',
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)