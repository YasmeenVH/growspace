import turtle
from random import randint
screen = turtle.Screen()
width = 84
height = 84
screen.setup(width=width, height=height)  # start position is at center of window
screen.bgcolor('black')
screen.tracer(0)



# Paddle
paddle = turtle.Turtle()    # Create a turtle object
paddle.shape('square')      # Select a square shape
paddle.speed(0)
paddle.shapesize(stretch_wid=1/20, stretch_len=1)   # Streach the length of square by 5
paddle.penup()
paddle.color('yellow')       # Set the color to white
paddle.goto(0, 42)        # Place the shape on bottom of the screen
print(paddle.get_shapepoly())

# Ball
ball = turtle.Turtle()      # Create a turtle object
ball.speed(0)
ball.shape('circle')        # Select a circle shape
ball.color('red')           # Set the color to red
ball.shapesize(1/20)
ball.penup()
ball.goto(-20, 30)           # Place the shape in middle

screen.update()
# PLANTBOT
#plant = turtle.Turtle()
#self.plant.pensize(10)  # set width larger
#self.plant.color('green')
#self.plant.penup()
#self.plant.goto(randint(-42, 42), -42)  # y must be 0