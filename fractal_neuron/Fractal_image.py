import numpy as np
from matplotlib import pyplot as plt
plt.style.use('science')
from svg_turtle import SvgTurtle 
def tree(branchLen,t, l):
    if branchLen > l:
        t.forward(branchLen)
        t.right(40)
        tree(branchLen-25,t, l)
        t.left(40)
        tree(branchLen-25,t, l)
        t.left(40)
        tree(branchLen-25,t, l)
        t.right(40)
        t.backward(branchLen)

def main():
    #  t = turtle.Turtle()
    t = SvgTurtle(500,500)
    #  myWin = turtle.Screen()
    t.left(90)
    t.up()
    t.backward(100)
    t.down()
    t.color("green")
    t.width(3)
    tree(120,t, 10)
    t.color("green")
    t.width(3)
    #  tree(120,t, 20)
    #  t.color("black")
    #  t.width(4)
    tree(120,t, 100)
    t.save_as('FIG_4A_2.svg')
    #  myWin.exitonclick()
main()
#  fig, ax = plt.subplots(figsize = (8,8))






















#  plt.show()


