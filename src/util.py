import sys, os 
import subprocess

def get_window_size():
    """Return lines, cols
    """
    return list(map(int, subprocess.check_output(['stty', 'size']).decode().split()))

def out_of_bounds(pos,max_x,max_y):
    x, y = pos.x, pos.y
    return x < 0 or y < 0 or x >= max_x or y >= max_y

def log(s, file_name='errors.log'):
    with open(file_name, 'a+') as f:
        f.write(str(s) + '\n')

class Position(object):
    def __init__(self, x=None, y=None, direction=None):
        if direction is None:
            self.x = x 
            self.y = y 
        else:
            self.x, self.y = direction % 2, (direction + 1) % 2
            if direction <= 2:
                self.x *= -1
                self.y *= -1

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y) 

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)

class Grid(object):
    def __init__(self, rows, cols, default_val):
        self.rows = rows
        self.cols = cols
        self.array = [[default_val for j in range(self.cols)] for i in range(self.rows)]

    def __len__(self):
        return len(self.array)

    def __getitem__(self, position):
        if isinstance(position, int):
            return self.array[position]
        return self.array[position.x][position.y]

    def __setitem__(self, position, val):
        self.array[position.x][position.y] = val

    def __str__(self):
        s = ''
        for r in self.array:
            s += ''.join(['% 3s' % str(c) for c in r]) + '\n'
        return s


