import gym
from random import randint
from gym.utils import seeding
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from growspace.plants.tree import branch

class GrowSpaceEnv(gym.Env):

    def __init__(self, observe_images, width = 84, height = 84, light_dif = 600):
        self.width = width         # do we keep?
        self.height = height       # do we keep?
        self.seed()
        self.images = observe_images
        self.light_dif = light_dif
        self.action_space = gym.spaces.Discrete(2)    # L, R of light paddle
        self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)

        self.x = np.random.randint(0,100)
        self.y = 0
        self.x2 = self.x
        self.y2 = 20
        self.branch = branch(self.x, self.x2, self.y, self.y2)   # initialize first upward branch

        self.x_target = np.random.randint(0,100)   # upper quadrant
        self.y_target = np.random.randint(80,100)
        self.light_width = 20
        self.x1_light = 40
        self.x2_light = self.x1_light + self.light_width
        self.fig = Figure
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.gca()

        self.x_scatter = np.random.randint(0,100, self.light_dif)
        self.y_scatter = np.random.randint(0 ,100, self.light_dif)

        self.width_plant = 1   # width of plant


    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def light_scatter(self):
        """ Input: x is the x-axis coordinate of the centroid position of the light source
            Output: 2D Scatter plot of the light (x,y) with density vector which increases with y"""
        d_coords = {k: v for v, k in enumerate(self.x_scatter)}
        idx_light = [d_coords[k] for k in np.arange(self.x1_light,self.x2_light+1 , 1)]
        x = [self.x_scatter[idx] for idx in idx_light]
        y = [self.y_scatter[idx] for idx in idx_light]

        return x, y

    def light_move_R(self):
        if self.x1_light >= 80:        # limit of coordinates
            self.x1_light = 80         # stay put

        else:
            self.x1_light += 10        # move by 10

    def light_move_L(self):
        if self.x1_light <= 10:        # limit of coordinates
            self.x1_light = 0
        else:
            self.x1_light -= 10        # move by 10

    def update_width(self):
        width = 0
        for i in range(len(self.child)):
            width += self.child[i].update_width()
        if width > 0:
            self.width_plant = width
        return self.width_plant

    def draw(self):
        self.ax.plot([self.x,self.x2],[self.y,self.y2], linewidth=np.sqrt(self.width_plant), color='green')

    def branch(self):


    def tree_grow(self, x, y, mindist, maxdist, branches):

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

        # increase thickness of first elements added to tree as they grow
        branches[0].updateWidth()

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
        # TARGET

        #self.plant = branch(self.x, self.x2, self.y, self.y2)

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
