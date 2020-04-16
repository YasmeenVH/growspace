# growspace
The goal of this project is to model plant branching with respect to light. The project presents an environment with a light source, a target and a plant growing.
The environment allows for the light to move on highest edge of the environment along the x-axis in order to maximize plant growth and guide the plant towards the desired target.

## Plant Growth Rules
The growth of the plant follows [L-systems](https://en.wikipedia.org/wiki/L-system) which have been widely used for recursive patterns such as fractals. L-systems were developped by Aristid Lindenmayer, a theoretical botanist, in a grammatical manner where branch segments are defined by string variables which are rewritten over time. 
In this environment, the main axiom will be identified as F (as forward)

### Example of simple L-system rules
ADD EXAMPLE

## Environment
The environment is a 84x84 pixel space where the light is located at the top of the frame and can move along the x-axis. The taret is located in the top third area of the environment and the plant always starts at a random location where the x-axis = 0. 
At each growing step the plant will 
### States

### Actions
- keep growing in the same direction
- change direction of growth
- stay dormant
- branch 

### Rewards 

## Installation
