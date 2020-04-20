import gym
import turtle
import numpy
from gym import spaces
from turtle import *
import tkinter as tk


class GrowSpaceEnv(gym.Env):

    def __init__(self, observe_images):
        self.images = observe_images
        self.action_space = gym.spaces.Discrete(2)

        if self.images:
            self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(0, 1, shape=(4,), dtype=np.float32)

        self.nodes = []
        self.leafs = []
        self.edges = []
        self.last_node_id = -1

        # BACKGROUND
        self.screen = turtle.Screen()
        self.screen.setup(width=84,height=84)                               # start position is at center of window
        self.screen.bgcolor('black')
        self.screen.tacer(0)

        # LIGHT
        self.light = turtle.Turtle()
        self.light.speed(0)
        self.light.shape('square')
        self.light.color('yellow')
        self.light.shapesize(stretch_wid=1, stretch_len=3)
        self.light.penup()
        self.light.goto(randint(-40, 40), 40)                                # since we are in a 84 by 84 0,0 is 42,42

        # TARGET
        self.target= turtle.Turtle()
        self.target.speed(0)
        self.target.shape('circle')
        self.target.color('red')
        self.target.penup()
        self.target.goto(randint(-40,40),randint(24,40))                      # upper quarter
        self.target.dx = 2
        self.target.dx = -2

        # PLANTBOT
        self.plantbot = turtle.Turtle()
        self.plantbot.pensize(10)                                               # set width larger
        self.plantbot.color('green')
        self.plantbot.penup()
        self.plantbot.goto(randint(-42,42),-42)                                 # y must be 0

    def get_observation(self):
        # do NOT make any new leafs
        # only show the image/state
        # if image, redraw from scratch

        if self.images:
            # TODO render an image of the current plant+light
            # you wanna return this as numpy 84x84x3 np.uint8 [0,255]

            for from_, to in self.edges:
                xx = x
        # TODO get coordinates of "from" node
        # TODO get coordinates of "to" node/leaf
        # TODO give a fat pencil to a brand new turtle and have it run from coordinates A to coordinates B in a straight line, then remove that turtle

        # TODO draw light source
        # TODO draw goal

        #else:
        # TODO return goal coordinates and coordinates of closest leaf
        # return this as np.array()

        # example: np.array([a,b,c,d]), where
        # a = goal_x
        # b = goal_y, can be fixed
        # c = closest_leaf_x # can be 0 when plant is freshly reset
        # d = closest_leaf_y # can be 0 when plant is freshly reset

        # TODO don't forget to normalize output (e.g. divide coordinates by 84 to move them into range [0,1])

        #pass
        return

    def reset(self):
        # TODO set the environment back to 0
        # (remove plant/start new plant,
        # reset light back to either 0 or random position,
        # reset turtle to start, etc.)
        # also pick new random goal

        self.light.goto(randint(-40, 40), 40)
        self.target.goto(randint(-40, 40), randint(24, 40))
        light_x = self.light.xcor()
        light_y = self.light.ycor()
        target_x = self.target.xcor()
        target_y = self.target.ycor()

           #return light_x, light_y, target_x, target_y

        return self.get_observation()

    def grow_branch(self, leaf):
        # TODO apply splitting/growth formula
        # based on growth formula, calculate new leaf coordinates
        # calculate 1 or 2 sets of x-y coordinates

        # this is where you use an invisible lil turtle
        # (only to calculate coordinates of leafs)

        # e.g. [(13,5),(15,6)]
        list_of_new_leafs = []
        for coord_x, coord_y in new_leafs:
            self.last_node_id += 1
            idx = self.last_node_id
            list_of_new_leafs.append(idx, coord_x, coord_y)

        return list_of_new_leafs

    def grow_plant(self):
        new_leafs = []

        for leaf in self.leafs:
            sprouts = self.grow_branch(leaf)
            new_leafs += sprouts

            for idx, coord_x, coord_y in sprouts:
                from_ = leaf[0]
                to = idx
                self.edges.append((from_, to))

        self.nodes += self.leafs
        self.leafs = new_leafs

    def step(self, action):
        # TODO sanitize action, make sure it's one of the two possible values

        # TODO apply action to environment:
        # move light based on action,
        # then make plant grow by one step <-- main important bit
        self.grow_plant()

        # TODO calculate reward:
        # find leaf that's closest to goal
        # calculate distance between goal and closest leaf
        # reward = -distance or 1/distance

        # TODO (optional) gather additional debugging infos and return the whole shebang

        observation = self.get_observation()
        reward = ...  # as calculated above
        done = False  # because we don't have a terminal condition
        misc = {}  # (optional) additional information about plant/episode/other stuff, leave empty for now

        return observation, reward, done, misc

    def render(self):
        self.screen.update()


if __name__=='__main___':
    b = turtle.Screen()
    b.title('Paddle')
    b.setup(width=84, height=84)  # start position is at center of window
    b.bgcolor('black')
    b.tracer(0)

    # LIGHT
    light = turtle.Turtle()
    light.speed(0)
    light.shape('square')
    light.color('yellow')
    light.shapesize(stretch_wid=1, stretch_len=3)
    light.penup()
    light.goto(randint(-40, 40), 40)  # since we are in a 84 by 84 0,0 is 42,42

    # TARGET
    target = turtle.Turtle()
    target.speed(0)
    target.shape('circle')
    target.color('red')
    target.penup()
    target.goto(randint(-40, 40), randint(24, 40))  # upper quarter
    target.dx = 2
    target.dx = -2
    turtle.done()


