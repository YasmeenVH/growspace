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
    id=f'GrowSpaceEnv-Control-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': None, 'setting': None,
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Control-Easy-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': None, 'setting': 'easy',
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Control-v0-hard',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': None, 'setting': 'hard',
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Control-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': 'Binary', 'level': None,
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Hierarchy-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': 'second',
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Hierarchy-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': 'Binary', 'level': 'second',
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Fairness-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Fairness',
    kwargs={'obs_type': None, 'level': None,
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Fairness-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Fairness',
    kwargs={'obs_type': 'Binary', 'level': 'second',
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Mnist-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Mnist',
    kwargs={'obs_type': None, 'level': None,
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Mnist-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Mnist',
    kwargs={'obs_type': 'Binary', 'level': None,
        # "observe_images": True
    },
    max_episode_steps=50, # number of total steps in the environment
)