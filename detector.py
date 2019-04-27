from numpy import *
from pylab import *
from random import randint
from math import * 

from lines import *
from neuron import neuron
from rules import *
from spike_train import *

T = 100
dt = .1

dimen = 10

i_inc = .00725
v_decay = .002

neurons = [
    [[[neuron() for x in range(dimen)] for y in range(dimen)] for z in range(dimen)],
    [neuron() for x in range(dimen)],
]

weights = [
    full((dimen, dimen, dimen), 1)
]

times = arange(0, T + dt, dt)

for i, net in enumerate(neurons[0]):
    for neuron in net[i]:
        neuron.on = True


def integrate(i, layer, st):
    for j, n in enumerate(neurons[layer]):
        prev_st = st[layer - 1][j]
        prev_weights = weights[layer - 1][j]

        res = 0
        for row_st, row_w  in zip(prev_st, prev_weights):
            # print len(row_st), len(row_w)
            res += dot(row_st[:, i], row_w) * i_inc
        
        # print i, j, n.v, res

        n.v += res

        if n.v > n.v_rest:
            n.v -= v_decay
    
    for j, n in enumerate(neurons[layer]):
        res = n.is_spike()
        if res:
            n.t_rest = times[i] + n.t_delay
            n.v = n.v_rest
            st[layer][j][i] = 1

def run(input):
    st = [[[[None for y in range(dimen)] for x in range(dimen)] for w in range(dimen)]]

    for i, net in enumerate(neurons[0]):
        test = 0
        for j, row in enumerate(net):
            for k, x in enumerate(row):
                curr = input[j][k]
                if bool(curr) == x.on:
                    st[0][i][j][k] = array(gen_st(1, len(times)))
                    test += 1
                else:
                    st[0][i][j][k] = array(gen_st(0, len(times)))
            
            st[0][i][j] = array(st[0][i][j])

        # print test

    st.append([zeros(len(times)) for x in range(dimen)])
    for i, t in enumerate(times):
        integrate(i, 1, st)
    
    return st


def report(st):
    res = [sum(out_st) for out_st in st[1]]
    return " & ".join([str(x) for x in res]) + " \\\\"

        
def main():
    for angle in range(0, 180, 10):
        dx = 1
        dy = tan(radians(angle))

        if dy != 0 and abs(dy) < 1:
            scale = 1 / abs(dy)
            dy *= scale
            dx *= scale
        
        dx = int(round(dx))
        dy = int(round(dy))

        input = gen_line(dx, dy, dimen)

        st = run(input)

        print(str(angle) + " & " + report(st)) # , dx, dy, tan(radians(angle))

        
        '''for i in range(dimen):
            print input[i]
        
        print "-----------"'''
        

        

    # report(st)

        # pots.append([[x.v for x in layer] for layer in neurons])


main()