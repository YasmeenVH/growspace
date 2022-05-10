Table of Contents
=================

<!--ts-->
* [Growspace](#growspace)
  * [Installation](#installation)
  * [Plant Branching](#plant-branching)
    * [Growth Algorithm Pseudocode](#growth-algorithm-pseudocode)
  * [Environment](#environment)
    * [Observations &  States](#observations----states)
    * [Actions](#actions)
      * [Light Movement](#light-movement)
      * [Light Focus](#light-focus)
    * [Rewards](#rewards)
    * [Challenges](#challenges)
      * [Control](#control)
      * [Hierarchical Learning](#hierarchical-learning)
      * [Fairness](#fairness)
      * [Multi-objective](#multi-objective)
  * [Training](#training)
<!--te-->

Growspace
============

The goal of this project is to model plant branching with respect to light. The project presents an environment with a light source, a target and a plant growing.The environment allows for the light to move on highest edge of the environment along the x-axis in order to maximize plant growth and guide the plant towards the desired target.

![alt text](https://github.com/YasmeenVH/growspace/blob/master/scripts/GrowSpaceEnv-HierarchyHard-v0-210218-163806.gif)


## Installation
``` python
# create conda environment with dependencies
conda env create --name envname -f ./scripts/conda/growspace.yml -y
conda activate conda_growspace
```

```
# for using jupyter-lab with conda env
conda install ipykernel
ipython kernel install --user --name=conda_growspace
```

```
git clone https://github.com/YasmeenVH/growspace
cd growspace
pip install -e .
cd ..
```

```
# if you prefer virtualenv over conda...
python -m pip install virtualenv
python -m virtualenv PATH_TO_VENV
source PATH_TO_VENV/bin/acitvate
pip install -r requirements.txt
```


## Plant Branching
The growth of the plant follows [Space Colonization Algorithm ](http://algorithmicbotany.org/papers/colonization.egwnp2007.large.pdf) which have been used for rending realistic trees in games. This algorithm is based on a cloud of points which have been inspired by the grown of tree in order to provide a certain attraction to the growing branches. 

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
The environment is a 84x84x3 pixel space where the light makes scattering points available. The target is located in the top third area of the environment and the plant always starts at a random location. The plant will only be attracted to current scattering under the light. The environment was made within the OpenAI Gym Environment framework.
### Observations &  States
Current position of branches and light position. Two types of observations are available: RGB images or stacked binary matrices representing the growing plant, the light and, the target.

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
env = gym.make('GrowSpaceEnv-Hierarchical-v0')
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

``` python
# PyTorch
conda install pytorch torchvision -c soumith

# Download Baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt

# Run Code
python main.py --env-name "GrowSpaceEnv-ControlEasy-v0" --custom-gym growspace --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 2500 --num-mini-batch 32 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --ppo-epoch 4

```


