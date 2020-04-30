import gym
from gym.utils import seeding
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from growspace.plants.tree import branch
from scipy.spatial import distance
import matplotlib.pyplot as plt

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
        #self.branch = self.branch()  # may need to add coords
        #self.branches = [branch(self.x, self.x2, self.y, self.y2)]  # initialize first upward branch

        self.branches = [branch(x =self.x, x2 = self.x2, y = self.y, y2 = self.y2)]
        self.x_target = np.random.randint(0,100)   # upper quadrant
        self.y_target = np.random.randint(80,100)
        self.light_width = 20
        self.x1_light = 40
        self.x2_light = self.x1_light + self.light_width

        self.x_scatter = np.random.randint(0,100, self.light_dif)
        self.y_scatter = np.random.randint(0,100, self.light_dif)


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

    def tree_grow(self, x, y, mindist, maxdist):

        # available scatter due to light position (look through xcoordinates
        for i in range(len(x) - 1, 0, -1):
            closest_branch = 0
            dist =  100

            # loop through branches and see which coordinate within available scatter is the closest
            for j in range(len(self.branches)):
                temp_dist = np.sqrt((x[i] - self.branches[j].x2) ** 2 + (y[i] - self.branches[j].y2) ** 2)  # could put in euclidean distance
                if temp_dist < dist:
                    dist = temp_dist
                    closest_branch = j

            # removes scatter points if reached
            if dist < mindist:
                x = np.delete(x, i)
                y = np.delete(y, i)

            # when distance is greater than max distance, branching occurs to find other points.
            elif dist < maxdist:
                self.branches[closest_branch].grow_count += 1
                self.branches[closest_branch].grow_x += (x[i] - self.branches[closest_branch].x2) / dist
                self.branches[closest_branch].grow_y += (y[i] - self.branches[closest_branch].y2) / dist

        # generation of new branches (forking) in previous step will generate a new branch with grow count
        for i in range(len(self.branches)):
            if self.branches[i].grow_count > 0:
                newBranch = self.branch(self.branches[i].x2, self.branches[i].x2 + self.branches[i].grow_x / self.branches[i].grow_count,
                                   self.branches[i].y2, self.branches[i].y2 + self.branches[i].grow_y / self.branches[i].grow_count)
                self.branches.append(newBranch)
                self.branches[i].child.append(newBranch)
                self.branches[i].grow_count = 0
                self.branches[i].grow_x = 0
                self.branches[i].grow_y = 0

        # increase thickness of first elements added to tree as they grow
        self.branches[0].update_width()
        branch_coords = []

        #sending coordinates out
        for i in range(len(self.branches)):
            branch_coords.append([self.branches[i].x2, self.branches[i].y2])  # x2 and y2 since they are the tips

        return branch_coords

    def distance_target(self, coords):
        # Calculate distance from each tip grown
        target_coord = [[self.x_target, self.y_target]]
        dist = distance.cdist(coords, target_coord, 'euclidean')

        # Get smallest distance to target
        min_dist = min(dist)

        return min_dist

    def get_observation(self):
        # only show the image/state
        plt.xlim(0,100)             # set axis limit, same as in branch()
        plt.ylim(0,100)
        plt.scatter(self.x_target, self.y_target, color = 'r')       # place target
        plt.axvspan(self.x1_light, self.x2_light, facecolor='y', alpha=0.5)   # place light

        # Draw plant
        for i in range(len(self.branches)):
            self.branches[i].draw_branch()

        # Make into image
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')

        # draw current state onto canvas
        canvas.draw()

        # image of observation as np array
        image = np.fromstring(canvas.tostring_rgb(), dtype = 'uint8')

        return image

    def reset(self):
        # Set env back to start
        self.x = np.random.randint(0,100)
        self.y = 0
        self.x2 = self.x
        self.y2 = 20
        self.branches = [branch( x = self.x, x2 = self.x2, y = self.y, y2 = self.y2)]  # initialize first upward branch

        self.x_target = np.random.randint(0,100)   # upper quadrant
        self.y_target = np.random.randint(80,100)
        self.light_width = 20
        self.x1_light = 40
        self.x2_light = self.x1_light + self.light_width

        self.x_scatter = np.random.randint(0,100, self.light_dif)
        self.y_scatter = np.random.randint(0 ,100, self.light_dif)

        return self.get_observation()

    def step(self, action):
        # Two possible actions, move light left or right
        if action == 0:
            self.light_move_R()

        if action == 1:
            self.light_move_L()

        # scattering that is available based on light's positions
        xx, yy = self.light_scatter()

        # Branching step for light in this position
        mindist = 1
        maxdist = 10
        tips = self.tree_grow(xx,yy,mindist,maxdist)

        # Calculate distance to target
        reward = 1/self.distance_target(tips)

        # Render image of environment at current state
        observation = self.get_observation() #image

        done = False  # because we don't have a terminal condition
        misc = {}  # (optional) additional information about plant/episode/other stuff, leave empty for now

        return observation, reward, done, misc

    def render(self, mode ='human'):
        pass


#if __name__=='__main__':

