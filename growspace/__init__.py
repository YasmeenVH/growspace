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
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-ControlEasy-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': None, 'setting': 'easy',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-ControlHard-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': None, 'setting': 'hard',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Control-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': 'Binary', 'level': None,
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-ControlEasy-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': 'binary', 'level': None, 'setting': 'easy',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)
register(
    id=f'GrowSpaceEnv-ControlHard-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': 'binary', 'level': None, 'setting': 'hard',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Hierarchy-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': 'second', 'setting': None,
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-HierarchyEasy-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': 'second', 'setting': 'easy',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-HierarchyHard-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': None, 'level': 'second', 'setting': 'hard',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Hierarchy-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': 'Binary', 'level': 'second', 'setting': None,
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-HierarchyEasy-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': 'binary', 'level': 'second', 'setting': 'easy',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-HierarchyHard-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Control',
    kwargs={'obs_type': 'binary', 'level': 'second', 'setting': 'hard',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Fairness-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Fairness',
    kwargs={'obs_type': None, 'level': None, 'setting': None,
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-FairnessEasy-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Fairness',
    kwargs={'obs_type': None, 'level': None, 'setting': 'easy',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-FairnessMiddle-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Fairness',
    kwargs={'obs_type': None, 'level': None, 'setting': 'hard_middle',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-FairnessAbove-v0',
    entry_point='growspace.envs:GrowSpaceEnv_Fairness',
    kwargs={'obs_type': None, 'level': None, 'setting': 'hard_above',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceEnv-Fairness-v1',
    entry_point='growspace.envs:GrowSpaceEnv_Fairness',
    kwargs={'obs_type': 'Binary', 'level': 'second',
            # "observe_images": True
            },
    max_episode_steps=50,  # number of total steps in the environment
)

register(
    id=f'GrowSpaceSpotlight-Mnist0-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '0',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist1-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '1',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist2-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '2',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist3-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '3',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist4-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '4',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist5-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '5',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist6-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '6',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist7-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '7',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist8-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '8',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist9-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '9',
            },
)

register(
    id=f'GrowSpaceSpotlight-Mnist1and7-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': '1_7_mix',
            },
)

register(
    id=f'GrowSpaceSpotlight-MnistMix-v0',
    entry_point='growspace.envs:GrowSpaceEnvSpotlightMnist',
    max_episode_steps=50,  # number of total steps in the environment
    kwargs={'digit': 'partymix',
            },
)
# register(
#     id=f'GrowSpaceEnv-Mnist-v1',
#     entry_point='growspace.envs:GrowSpaceEnv_Mnist',
#     kwargs={'obs_type': 'Binary', 'level': None,
#             # "observe_images": True
#             },
#     max_episode_steps=50,  # number of total steps in the environment
# )

register(
    id=f'GrowSpaceEnv-Continuous-v0',
    entry_point='growspace.envs:GrowSpaceContinuous',
    max_episode_steps=50,
)

# TOOD(Manuel): Register water
