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

def gen_square(top_left, n, dimen):
    res = [[0 for x in range(dimen)] for y in range(dimen)]

    i = top_left[0]
    while (i - top_left[0]) < n:
        j = top_left[1]

        while (j - top_left[1]) < n:
            res[i][j] = 1
            j += 1

        i += 1

    
    return res
