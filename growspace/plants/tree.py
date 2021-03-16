import numpy as np

GROWTH_MULTIPLIER = .02  # keep this below 1 to prevent exponential growth



def valid_coord(x):
    assert isinstance(x, int)
    assert 0 <= x < 71
    return True


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
            width += self.child[i].update_width()  # because exponential growth

        self.width += width * GROWTH_MULTIPLIER

        return self.width

    def get_pt1_pt2(self):
        x1 = int(round(self.x * self.img_width))
        x2 = int(round(self.x2 * self.img_width))

        y1 = int(round(self.y * self.img_height))
        y2 = int(round(self.y2 * self.img_height))

        return (x1, y1), (x2, y2)

    # def draw_branch(self):
    #     plt.ylim(0, 100)
    #     plt.xlim(0, 100)
    #     plt.plot([self.x,self.x2],[self.y,self.y2], linewidth=np.sqrt(self.width), color='green')


class PixelBranch:
    def valid_coord(self, x):
        assert isinstance(x, int)
        assert 0 <= x < self.img_side
        return True

    def __init__(self, x, x2, y, y2, img_width, img_height):
        assert img_width == img_height, "Only square images are supported"
        self.img_side = img_width

        assert self.valid_coord(x)
        assert self.valid_coord(x2)
        assert self.valid_coord(y)
        assert self.valid_coord(y2)
        assert self.valid_coord(x)
        assert self.valid_coord(x2)
        assert self.valid_coord(y)
        assert self.valid_coord(y2)

        self.p = (x, y)
        self.tip_point = (x2, y2)

        self.grow_count = 0
        self.grow_direction = np.array([0, 0])
        self.width = .5
        self.child = []

    @property
    def x(self):
        return self.p[0]

    @property
    def y(self):
        return self.p[1]

    @property
    def x2(self):
        return self.tip_point[0]

    @property
    def y2(self):
        return self.tip_point[1]

    def update_width(self):
        width = 0
        for i in range(len(self.child)):
            width += self.child[i].update_width()  # because exponential growth

        self.width += width * GROWTH_MULTIPLIER

        return self.width


if __name__ == '__main__':
    branches = [Branch(30, 30, -10, 0, 84, 84)]
    print(len(branches))
    print(type(branches[0].get_pt1_pt2()))
