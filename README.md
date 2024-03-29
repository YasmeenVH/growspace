GrowSpace
=================

The goal of this project is to model plant branching with respect to light. The project presents an environment with a light source, a target and a plant growing. The environment allows for the light to move along the top edge of the environment (along the x-axis) in order to maximize plant growth and guide the plant towards the desired target.

![alt text](https://github.com/YasmeenVH/growspace/blob/master/scripts/GrowSpaceEnv-HierarchyHard-v0-210218-163806.gif)

<!--ts-->
 * [Installation](#installation)
  * [Plant Branching](#plant-branching)
    * [Growth Algorithm Pseudocode](#growth-algorithm-pseudocode)
  * [Environment](#environment)
    * [Observations & States](#observations----states)
    * [Actions](#actions)
    * [Rewards](#rewards)
    * [Challenges](#challenges)
  * [Training](#training)
<!--te-->

## Installation
Tested on:
- Ubuntu
- MacOS

``` python
# create conda environment with dependencies
conda env create -f growspace/environment.yml 
conda activate conda_growspace
```

``` python
# if you prefer virtualenv over conda...
python -m virtualenv PATH_TO_VIRTUALENV
source PATH_TO_VIRTUALENV/bin/activate
pip install -r growspace/requirements.txt # clone it first to run this git clone https://github.com/YasmeenVH/growspace
```

Install GrowSpace:
``` python
# install growspace with pip from the repo
git clone https://github.com/YasmeenVH/growspace
cd growspace
pip install -e .
cd ..
```

How to add conda/virtualenv to Jupyter:
``` python
# to add conda env to Jupyter
conda install ipykernel
ipython kernel install --name=conda_growspace

# to add python virtualenv to jupyter
pip install ipykernel
python -m ipykernel install --name=PATH_TO_VENV
```

The demo notebook `demo_growsapce_control.ipynb` shows some features of GrowSpace and how to run it with stable_baselines3 agents.

## Plant Branching
The algorithm used was the [Space Colonization Algorithm ](http://algorithmicbotany.org/papers/colonization.egwnp2007.large.pdf) which is based on leaf venation and has been used for rending realistic trees in games. 

![alt text](https://github.com/YasmeenVH/growspace/blob/master/scripts/beam.png)

### Growth Algorithm Pseudocode

```python
growth_len = val # max branch growth len
branches = []
min_dist = val # minimum distance under which a space colonization point is reached
max_dist = val # maximum distance under which a branch can grow towards a space colonization point  

def reset():
    points = fill_arena_with_random_points()     
    branch_start, branch_end = create_first_branch()
    branches.append(Branch(start=branch_start,end=branch_stop))
    light_left, light_right = position_light_source()

def step():
    points_filtered = point_range(points, light_left-growth_len, light_right+growth_len)
    branches_filtered = branch_range(branches, light_left, light_right)
    for point in points_filtered:
        closest_branch, dist = find_closest_branch(point, branches_filtered)
        if dist < min_dist:
            remove_point(point) # if we reach the point, we don't branch
        elif dist < max_dist:
            closest_branch.add_growth_vector_towards(point)
        
    branches_to_grow_from = find_branches_with_growths(branches_filtered)
    for branch in branches_to_grow_from:
        branches.append(Branch(start=branch.end, end=branch.end+branch.growth_vector * growth_len))

    update_all_branch_widths(branches)

```

## Environment
The environment is a 84x84x3 pixel space where the light makes scattering points available for branching. The target is located in the top third area of the environment and the plant always starts at a random location. The plant will only be attracted to scattering points under the light. The environment was built following the OpenAI Gym Environment framework.

### Observations & States
The agent has access to two types of observations; stacked binary matrices or RGB pictures. are in picture format. All resulsts with stable baselines has been done with RGB observations. The stacked binary matrices represent the light position, the plant position and the target position. 

### Actions
#### Light Movement
- Move light right
- Move light left
- Stay in current position 

#### Light Focus
- Increase light width
- Decrease light width

### Rewards 
Closest distance to the target(s)

### Challenges
 Tasks               | Control       | Hierarchical  | Fairness | Multi-Objective
---|---|---|---|--- 
 Grow Plant          |  [x]          | [x]           | [x]      | [x]            
 Get to Target       |  [x]          | [x]           | [x]      |                
 Find Plant          |               | [x]           | [x]      | [x]            
 Grow Multiple Plants|               |               | [x]      |                
 Grow into Shape     |               |               |          | [x]            


#### Control
Grow the plant to target with the light beam. An episode starts with the light above the plant.
```python
import gym
import growspace  
env = gym.make('GrowSpaceEnv-Control-v0')
```

#### Hierarchical Learning 
Find the plant with the light beam to initiliaze growth and grow the plant to target with the light beam.
```python
env = gym.make('GrowSpaceEnv-Hierarchy-v0')
```

#### Fairness
Find both plants and grow them towards the target. The objective is to maintain plants at similar growth stages for every time step.
```python
env = gym.make('GrowSpaceEnv-Fairness-v0')
```
#### Multi-objective 
```python
env = gym.make('GrowSpaceSpotlight-MnistMix-v0')
```

## Training
- Repository for testing PPO and A@C baselines is found [here](https://github.com/YasmeenVH/growspaceenv_baselines/tree/master/a2c_ppo_acktr)
- Repository for testing Rainbow baseline is found [here](https://github.com/manuel-delverme/rainbow_growspace)

#### Example of training with stable_baselines3 PPO

``` python
# example code below

from stable_baselines3 import PPO

env = gym.make('GrowSpaceEnv-Control-v0')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
```
