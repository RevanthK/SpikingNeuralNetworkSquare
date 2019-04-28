from numpy import *
from pylab import *
from random import randint
from math import * 

from neuron import neuron
from rules import *
from spike_train import *
from lines import gen_square

from sys import argv

iters = 100

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

square_sizes = [2, 4, 6]

def_size = 4 # default square size
dimen = 10

input_size = dimen ** 2

#TODO: Scale each layer by num of sizes 
neurons = [
    [neuron() for x in range(dimen**2)], # Input Layer 
    [neuron() for x in range((dimen - def_size) ** 2))], # Bounding Box Layer
    [neuron() for x in range(1)] # Output OR Gate
]

weights = gen_weights(dimen ** 2, neurons)

def gen_weights(input_size, neurons):
    weights = []
    weights.append(full((input_size, len(neurons[0])), 1))

    for i in range(1, len(neurons)):
        weights.append(full(len(neurons[i - 1], len(neurons[i])), 1))
    
    return weights


'''def layer_square(n, input):
    x_length = len(input[0])
    x_bound = x_length - n
    y_length = len(input)
    y_bound = y_length - n

    top_lefts = [[None for x in range(x_bound)] for y in range(y_bound)]

    for i in range(top_lefts):
        for j in range(top_lefts[0]):'''


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

            res_currents = multiply(prev_currents, st[layer - 1][:, i])

            dv = dot(weights[layer - 1][j], res_currents)

            n.v += dv

            if n.v > n.v_rest:
                n.v -= v_decay
    
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

            weights[layer - 1][j][k] += ojas(f, s, weights[layer - 1][j][k])


def train(inputs, sizes, update_weights=None, ans=None):
    inputs = flatten(inputs)

    for layer in neurons:
        for neuron in layer:
            neuron.clear()

    hidden_st = [[0 for x in times] for y in range(hidden_size)]
    
    st = [array([gen_st(x, len(times), mult) for x in inputs])]
    st += [zeros((1, len(x)) for x in neurons]

    input_currents = [zeros(len(times)) for x in range(input_size)]
    
    hidden_currents = [zeros(len(times)) for x in range(hidden_size)]
    
    currents = [
        full((2, len(times)), i_inc),
        array(input_currents),
        array(hidden_currents),
        zeros((output_size, len(times)))
    ]

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
    else:
        return

    print("Start Weights: ")
    print(weights[1])
    print(weights[2])
    
    # TODO: Train on diff locations
    for i in range(iters):
        st, currents, pots = train(gen_square([0, 0], 2, 10), update_rule, [1, 0, 0]) # 2
        st, currents, pots = train(gen_square([0, 0], 4, 10), update_rule, [0, 1, 0]) # 4
        st, currents, pots = train(gen_square([0, 0], 6, 10), update_rule, [0, 0, 1]) # 6


    print("After: ")
    print(weights[1])
    print(weights[2])



