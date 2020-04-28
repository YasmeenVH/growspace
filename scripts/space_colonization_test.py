import turtle
import numpy as np
import random
from random import randint

class branch():
    def __init__(self, x, x2, y, y2):
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.grow_count = 0
        self.grow_x = 0
        self.grow_y = 0
        self.width = 1
        self.child = []

        self.screen = turtle.Screen()
        self.screen.setup(width=84, height=84)
        self.screen.bgcolor('black')


        self.tree = turtle.Turtle()
        self.tree.hideturtle()
        self.tree.color('green')
        self.tree.speed(0)
        self.tree.pensize(2)



    def plot(self):
        self.tree.penup()
        #self.tree.hideturtle()

        self.tree.goto(self.x, self.y)  # make the turtle go to the start position
        self.tree.pendown()
        self.tree.goto(self.x2, self.y2)
        self.screen.update()

def draw(x, y, mindist, maxdist, branches):


    for i in range(len(x) - 1, 0, -1):
        closest_branch = 0
        dist = 109

        for j in range(len(branches)):
            temp_dist = np.sqrt((x[i] - branches[j].x2) ** 2 + (y[i] - branches[j].y2) ** 2)
            if temp_dist < dist:
                dist = temp_dist
                closest_branch = j

        # removes scatter
        if dist < mindist:
            x = np.delete(x, i)
            y = np.delete(y, i)

        elif dist < maxdist:
            branches[closest_branch].grow_count += 1
            branches[closest_branch].grow_x += (x[i] - branches[closest_branch].x2) / dist
            branches[closest_branch].grow_y += (y[i] - branches[closest_branch].y2) / dist

    for i in range(len(branches)):
        if branches[i].grow_count > 0:
            newBranch = branch(branches[i].x2, branches[i].x2 + branches[i].grow_x / branches[i].grow_count,
                               branches[i].y2, branches[i].y2 + branches[i].grow_y / branches[i].grow_count)
            branches.append(newBranch)
            branches[i].child.append(newBranch)
            branches[i].grow_count = 0
            branches[i].grow_x = 0
            branches[i].grow_y = 0


if __name__=='__main__':
    x = np.random.randint(-41, 41, 500)
    y = np.random.randint(-27, 41, 500)

    branches = [branch(0, 0, -42, -27)]
    #branch = branch()
    print(len(branches))
    branches[0].plot()

    maxdist = 6
    mindist = 1
    for d in range(100):
        draw(x, y, mindist, maxdist, branches )
        for i in range(len(branches)):
            branches[i].plot()
        #print('it is drawing')
