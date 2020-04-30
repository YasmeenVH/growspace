import numpy as np
import matplotlib.pyplot as plt

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

    def update_width(self):
        width = 0
        for i in range(len(self.child)):
            width += self.child[i].updateWidth()

        if width > 0:
            self.width = width

        return self.width

    def draw(self):
        plt.plot([self.x,self.x2],[self.y,self.y2], linewidth=np.sqrt(self.width), color='green')

