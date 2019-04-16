from numpy import *

def gen_line(dy, dx, dimen):
    res = zeros((dimen, dimen))

    pos = [dimen / 2 - 1,  dimen / 2 - 1]

    while pos[0] < dimen and pos[1] < dimen and pos[0] >= 0 and pos[1] >= 0:
        res[pos[0]][pos[1]] = 1

        pos[0] += dx
        pos[1] += dy
    
    pos = [dimen / 2 - 1,  dimen / 2 - 1]

    while pos[0] < dimen and pos[1] < dimen and pos[0] >= 0 and pos[1] >= 0:
        res[pos[0]][pos[1]] = 1

        pos[0] -= dx
        pos[1] -= dy
    
    return res

