GROWTH_MULTIPLIER = .1 # keep this below 1 to prevent exponential growth

class Branch(object):

    def __init__(self, x, x2, y, y2, img_width, img_height):
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2

        self.img_width = img_width
        self.img_height = img_height

        self.grow_count = 0
        self.grow_x = 0
        self.grow_y = 0
        self.width = .5
        self.child = []

    def update_width(self):
        width = 0
        for i in range(len(self.child)):
            width += self.child[i].update_width() # because exponential growth

        self.width += width * GROWTH_MULTIPLIER

        return self.width

    def get_pt1_pt2(self):
        x1 = int(round(self.x * self.img_width))
        x2 = int(round(self.x2 * self.img_width))

        y1 = int(round(self.y * self.img_height))
        y2 = int(round(self.y2 * self.img_height))

        return (x1,y1), (x2,y2)

    # def draw_branch(self):
    #     plt.ylim(0, 100)
    #     plt.xlim(0, 100)
    #     plt.plot([self.x,self.x2],[self.y,self.y2], linewidth=np.sqrt(self.width), color='green')

if __name__=='__main__':
    branches = [Branch(30, 30, -10, 0)]
    print(len(branches))
    branches[0].draw_branch()