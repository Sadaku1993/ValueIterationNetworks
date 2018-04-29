import random
import numpy as np
import math

size_row = 16
size_col = 16

size = (size_row, size_col)

goal = (random.randint(1, size_row-2), random.randint(1, size_col-2))

grid = np.zeros(size, dtype=np.uint8)

# goal
grid[goal[0], goal[1]] = 9

# wall
grid[:, 0] = 1
grid[:, -1] = 1
grid[0, :] = 1
grid[-1, :] = 1

# random obstacle
max_obstacle = 40
num_obstacle = random.randint(0, max_obstacle)
for i in xrange(num_obstacle):
    rand_row = random.randint(0, size_row-1)
    rand_col = random.randint(0, size_col-1)

    if (rand_row == goal[0] and rand_col == goal[1]):
        continue
    if (grid[rand_row][rand_col] == 1):
        continue
    else:
        grid[rand_row][rand_col] = 2

for i in xrange(size_row):
    for j in xrange(size_col):
        if (goal[0]==i and goal[1]==j):
            print("G"),
        elif (grid[i][j] == 0):
            print(" "),
        elif (grid[i][j] == 1):
            print("w"),
        else:
            print("*"),
    print (" ")
