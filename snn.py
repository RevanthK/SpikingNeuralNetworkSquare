from numpy import *
from pylab import *
from random import randint
from math import * 

from neuron import neuron
from rules import *
from spike_train import *

from sys import argv

iters = 100

input_size = 2
hidden_size = 2
output_size = 1

T = 25
dt = .1


i_inc = .5
i_decay = .001
i_max = .5

v_decay = .1

teach_current = 20

mult = 10

win = 20

log = False

weights = [
    full((input_size, 2), 1),
    [[1 for x in range(input_size)] + [0, 0] for x in range(hidden_size)],
    [[1 for x in range(hidden_size)] + [1] for x in range(output_size)]
]

neurons = [
    [neuron() for x in range(input_size)],
    [neuron() for x in range(hidden_size)],
    [neuron() for x in range(output_size)]
]

neurons[1][0].inhib = True # NAND gate



def layer_square(n, input):
    x_length = len(input[0])
    x_bound = x_length - n
    y_length = len(input)
    y_bound = y_length - n

    top_lefts = [[None for x in range(x_bound)] for y in range(y_bound)]

    for i in range(top_lefts):
        for j in range(top_lefts[0]):


times = arange(0, T + dt, dt)

# integrate even the first layer
def integrate(i, layer, st, currents, teach=False):
    for j, n in enumerate(neurons[layer - 1]):
        if n.t_rest < times[i]:

            if layer == 1:
                prev_currents = zeros(input_size)
                prev_currents[j] = i_inc
            else:
                prev_currents = currents[layer - 1][:, i]
            
            # print "cok", layer
            # print layer, len(prev_currents), len(st[layer - 1][:, i])

            res_currents = multiply(prev_currents, st[layer - 1][:, i])

            # print layer, res_currents, weights[layer - 1][j]


            dv = dot(weights[layer - 1][j], res_currents)

            if layer == 2 and teach:
                dv += prev_currents[input_size + j]
            elif layer == 2:
                # print "Layer 2 dv: ", dv
                pass

            if n.inhib:
                dv = i_max - dv
                # print "Inhib Dv: ", dv

            n.v += dv

            if n.v > n.v_rest:
                n.v -= v_decay

            if layer == 2 and dv > 0 and log:
                # print n.v, layer, j, dv, currents[layer][j][i - 1]
                pass
    
    for j, n in enumerate(neurons[layer -  1]):
        res = n.is_spike()
        currents[layer][j][i] =  currents[layer][j][i - 1]
        if res:
            # print "Spiked ", layer, j, i
            currents[layer][j][i] += i_inc

            n.t_rest = times[i] + n.t_delay
            n.v = n.v_rest
            st[layer][j][i] = 1
        
        if currents[layer][j][i] > 0:
            currents[layer][j][i] -= i_decay # maybe allow neg currents



def update_ojas(i, layer, st, neurons):
    for j, n in enumerate(neurons[layer - 1]):
        for k, f in enumerate(neurons[layer - 2]):
            f = st[layer - 1][k][i]
            s = st[layer][j][i]

            if layer == 3 and k == 1:
                # print f, s, rule(f, s, weights[layer - 1][j][k])
                pass

            # print f, s
            weights[layer - 1][j][k] += ojas(f, s, weights[layer - 1][j][k])

def update_stdp(i, layer, st, neurons):
    for j, x in enumerate(neurons[layer - 1]):
        curr_st = st[layer][j]
        if curr_st[i] != 1:
            continue

        for k, y in enumerate(neurons[layer - 2]):
            prev_st = st[layer - 1][k]

            for pos in range(i - win, i + win):
                if pos < 0 or pos >= len(times):
                    continue
                
                if prev_st[pos] == 1:
                    # print layer, j, k
                    weights[layer - 1][j][k] = stdp(weights[layer - 1][j][k], pos - i)


    
def train(inputs, update_weights=None, ans=None):
    for layer in neurons:
        for neuron in layer:
            neuron.clear()

    teach = (not ans is None)

    if (not ans is None):
        val = 1

        if ans[0]:
            nand_current = teach_current
        else:
            nand_current = -teach_current

        if ans[1]:
            or_current = teach_current
        else:
            or_current = -teach_current

        if ans[2]:
            xor_current = teach_current
        else:
            xor_current = -teach_current
    else:
        val = 0
        nand_current = 0
        or_current = 0
        xor_current = 0

    hidden_st = [[0 for x in times] for y in range(hidden_size)] + [full(len(times), val)]
    
    st = [
        array([gen_st(x, len(times), mult) for x in inputs]),
        array([[0 for x in times] for y in range(input_size)] + [[1 for x in times], [1 for x in times]]),
        array(hidden_st),
        array([[0 for x in times] for y in range(output_size)])
    ]


    input_currents = [zeros(len(times)) for x in range(input_size)]
    
    input_currents += [full(len(times), nand_current)]
    input_currents += [full(len(times), or_current)]
    
    hidden_currents = [zeros(len(times)) for x in range(hidden_size)] + [full(len(times), teach_current)]
    
    currents = [
        full((2, len(times)), i_inc),
        array(input_currents),
        array(hidden_currents),
        zeros((output_size, len(times)))
    ]

    # print currents[1]

    # print "Length: ", len(st[1]), len(st[1][:, 0])

    pots = []

    for i, t in enumerate(times):
        integrate(i, 1, st, currents, teach)
        integrate(i, 2, st, currents, teach)
        integrate(i, 3, st, currents, teach)

        pots.append([[x.v for x in layer] for layer in neurons])

        if (not ans is None):
            update_weights(i, 2, st, neurons)
            update_weights(i, 3, st, neurons)
    
    return st, currents, pots

def report(st, currents, pots):
    print("Output: ", sum(st[3][0]))
    print("ST Fires: ", [sum(x) / mult for x in st[0]])
    print("Input Fires: ", [sum(x) for x in st[1]])
    print("Hidden Fires: ", [sum(x) for x in st[2]])

    print("Input Currents: ", [x[-1] for x in currents[1]])
    print("Hidden Currents: ", [x[-1] for x in currents[2]])

    print("Input Pots: ", pots[-1][1])
    print("Hidden Pots: ", pots[-1][2])
    print("-----------------------")


def main():
    if int(argv[1]) == 0:
        update_rule = update_ojas
    elif int(argv[1]) == 1:
        update_rule = update_stdp
    else:
        return

    print("Start Weights: ")
    print(weights[1])
    print(weights[2])

    for i in range(iters):
        st, currents, pots = train([1, 1], update_rule, [0, 1, 0])
        st, currents, pots = train([0, 1], update_rule, [1, 1, 1])
        st, currents, pots = train([1, 0], update_rule, [1, 1, 1])
        st, currents, pots = train([0, 0], update_rule, [1, 0, 0])

        # print [x.v for x in neurons[1]]

    print("After: ")
    print(weights[1])
    print(weights[2])

    global log
    log = True
    
    '''weights[2][0][0] = 0
    weights[2][0][1] = 0
    weights[2][0][2] = 0'''
    
    print("0 0")
    st, currents, pots = train([0, 0]) # test
    report(st, currents, pots)

    print("0 1")
    st, currents, pots = train([0, 1]) # test
    report(st, currents, pots)

    print("1 0")
    st, currents, pots = train([1, 0]) # test
    report(st, currents, pots)

    print("1 1")
    st, currents, pots = train([1, 1]) # test
    report(st, currents, pots)

main()

