GROWTH_MULTIPLIER = .02  # keep this below 1 to prevent exponential growth


def valid_coord(x):
    assert isinstance(x, int)
    assert 0 <= x < 71
    return True


class Branch(object):

    def __init__(self, x, x2, y, y2, img_width, img_height):
        assert valid_coord(x)
        assert valid_coord(x2)
        assert valid_coord(y)
        assert valid_coord(y2)
        assert valid_coord(x)
        assert valid_coord(x2)
        assert valid_coord(y)
        assert valid_coord(y2)
        self.p = (x, y)
        self.p2 = (x2, y2)

        self.img_width = img_width
        self.img_height = img_height

        self.grow_count = 0
        self.grow_x = 0
        self.grow_y = 0
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
        return self.p2[0]

    @property
    def y2(self):
        return self.p2[1]

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


if __name__ == '__main__':
    branches = [Branch(30, 30, -10, 0, 84, 84)]
    print(len(branches))
    print(type(branches[0].get_pt1_pt2()))
