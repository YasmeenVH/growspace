import turtle


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

        #self.screen = turtle.Screen()
        #self.screen.setup(width=84, height=84)
        #elf.screen.bgcolor('black')
        self.tree = turtle.Turtle()
        self.tree.hideturtle()
        self.tree.color('green')
        self.tree.speed(0)
        self.tree.pensize(2)



    def plot(self):

        self.tree.penup()
        self.tree.goto(self.x, self.y)  # make the turtle go to the start position
        self.tree.pendown()
        self.tree.goto(self.x2, self.y2)
        self.screen.update()