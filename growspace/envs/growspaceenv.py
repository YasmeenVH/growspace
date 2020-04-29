import gym
import turtle
import random
from random import randint
from gym.utils import seeding
import numpy as np
from growspace.plants.tree import branch


class GrowSpaceEnv(gym.Env):

    def __init__(self, observe_images, width = 84, height = 84, light_dif = 500):
        self.width = width
        self.height = height
        self.seed()
        self.images = observe_images
        self.light_dif = light_dif
        self.action_space = gym.spaces.Discrete(2)    # L, R of light paddle

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
        self.screen.setup(width= width,height= height)                               # start position is at center of window
        self.screen.bgcolor('black')
        self.screen.tracer(0)

        # LIGHT
        self.light = turtle.Turtle()
        self.light.speed(0)
        self.light.shape('square')
        self.light.color('yellow')
        self.light.shapesize(stretch_wid=1/20, stretch_len=1)

        # TARGET
        self.target= turtle.Turtle()
        self.target.speed(0)
        self.target.shape('circle')
        self.target.hideturtle()
        self.target.shapesize(1/20)
        self.target.color('red')

        # TREEBOT
        self.x = np.random.randint(-41,42)
        self.y = -42
        self.x2 = self.x
        self.y2 = -27
        self.plant = branch(self.x, self.x2, self.y, self.y2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def light_scatter(self):
        """ Input: x is the x-axis coordinate of the centroid position of the light source
            Output: 2D Scatter plot of the light (x,y) with density vector which increases with y"""
        x_scatter = np.random.randint(-41,41, self.light_dif)
        y_scatter = np.random.randint(-27 ,41, self.light_dif)
        coords = []
        for i in range(len(x_scatter)):
            one_coord = [x_scatter[i],y_scatter[i]]
            coords.append(one_coord)

        return coords

    def light_move_R(self):
        """Move by 5 pixels in the right direction"""
        x = self.light.xcor()
        if x >= 42-5:
            self.light.setx(x)
        else:
            self.light.setx(x+5)

    def light_move_L(self):
        """Move by 5 pixels in the left direction"""
        x = self.light.xcor()
        if x <= -42-5:
            self.light.setx(x)
        else:
            self.light.setx(x-5)


    def scatter_with_light(self, coords, light_x):
        #define light posiiton
        #xx, yy = light_coord
        possible_branch = []
        x = self.light.xcor()
        x1 = x-5 # left boundary of light
        x2 = x+5 # right boundary of light
        for x, y in coords:
            if x1 <= x <= x2:
                possible_branch.append(coords)
            for xx, yy in light_coords:
                if

    def space_colonization(self, x, y, mindist, maxdist, branches):

        # available scatter due to light position (look through xcoordinates
        for i in range(len(x) - 1, 0, -1):
            closest_branch = 0
            dist =  100

            # loop through branches and see which coordinate within available scatter is the closest
            for j in range(len(branches)):
                temp_dist = np.sqrt((x[i] - branches[j].x2) ** 2 + (y[i] - branches[j].y2) ** 2)
                if temp_dist < dist:
                    dist = temp_dist
                    closest_branch = j

            # removes scatter points if reached
            if dist < mindist:
                x = np.delete(x, i)
                y = np.delete(y, i)

            # when distance is greater than max distance, branching occurs to find other points.
            elif dist < maxdist:
                branches[closest_branch].grow_count += 1
                branches[closest_branch].grow_x += (x[i] - branches[closest_branch].x2) / dist
                branches[closest_branch].grow_y += (y[i] - branches[closest_branch].y2) / dist

        # generation of new branches (forking) in previous step will generate a new branch with grow count
        for i in range(len(branches)):
            if branches[i].grow_count > 0:
                newBranch = branch(branches[i].x2, branches[i].x2 + branches[i].grow_x / branches[i].grow_count,
                                   branches[i].y2, branches[i].y2 + branches[i].grow_y / branches[i].grow_count)
                branches.append(newBranch)
                branches[i].child.append(newBranch)
                branches[i].grow_count = 0
                branches[i].grow_x = 0
                branches[i].grow_y = 0

    def get_observation(self):
        # do NOT make any new leafs
        # only show the image/state
        # if image, redraw from scratch

        if self.images:
            # TODO render an image of the current plant+light
            # you wanna return this as numpy 84x84x3 np.uint8 [0,255]

            for from_, to in self.edges:
                xx = self.edges

        # TODO get coordinates of "from" node
        # TODO get coordinates of "to" node/leaf
        # TODO give a fat pencil to a brand new turtle and have it run from coordinates A to coordinates B in a straight line, then remove that turtle

        # TODO draw light source
        # TODO draw goal

        else:

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
        self.light.penup()
        self.light.goto(randint(-41, 41), 40)
        self.target.penup()
        self.target.goto(randint(-40, 40), randint(24, 40)) # upper quarter
        #self.plant.clear()       # removes plant drawing
        #self.plant.goto()

        light_x = self.light.xcor()
        light_y = self.light.ycor()
        target_x = self.target.xcor()
        target_y = self.target.ycor()

           #return light_x, light_y, target_x, target_y

        return self.get_observation()

    #def grow_branch(self, leaf):
        # TODO apply splitting/growth formula
        # based on growth formula, calculate new leaf coordinates
        # calculate 1 or 2 sets of x-y coordinates

        # this is where you use an invisible lil turtle
        # (only to calculate coordinates of leafs)

        # e.g. [(13,5),(15,6)]
        #list_of_new_leafs = []
        #for coord_x, coord_y in new_leafs:
            #self.last_node_id += 1
            #idx = self.last_node_id
            #list_of_new_leafs.append(idx, coord_x, coord_y)

        #return list_of_new_leafs

    #def grow_plant(self):
        #new_leafs = []

        #for leaf in self.leafs:
            #sprouts = self.grow_branch(leaf)
            #new_leafs += sprouts

            #for idx, coord_x, coord_y in sprouts:
                #from_ = leaf[0]
                #to = idx
                #self.edges.append((from_, to))

        #self.nodes += self.leafs
        #self.leafs = new_leafs

    def step(self, action):
        # TODO sanitize action, make sure it's one of the two possible values
        if action == 0:
            self.light_move_R()

        if action
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

    def render(self, mode ='human'):
        self.screen.update()


if __name__=='__main__':


