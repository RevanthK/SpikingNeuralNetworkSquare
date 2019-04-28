from numpy import *
from pylab import *
from random import randint
from math import * 

from neuron import neuron
from rules import *
from spike_train import *
from lines import gen_square

from sys import argv

iters = 10 # How many times to train

T = 25
dt = .1


i_inc = .5 # current delta for spike
i_decay = .001
i_max = .5 

v_decay = .1

teach_current = 20

mult = 10 # spike value in spike train

log = False

square_sizes = [4]

def_size = 4 # default square size
dimen = 10

input_size = dimen ** 2
 
neurons = [
    [neuron() for x in range(input_size * len(square_sizes))], # Input Layer 
    [neuron() for x in range(sum([(dimen - sq_size) ** 2 for sq_size in square_sizes]))], # Bounding Box Layer
    [neuron() for x in range(len(square_sizes))] # Output OR Gate
]

def gen_weights(input_size, neurons):
    weights = []
    weights.append(full((input_size, len(neurons[0])), 1))

    for i in range(1, len(neurons)):
        weights.append(full((len(neurons[i]), len(neurons[i - 1])), 1))
        # print weights[i].shape
    
    return weights

weights = gen_weights(dimen ** 2, neurons)

times = arange(0, T + dt, dt)

# integrate even the first layer
def integrate(i, layer, st, currents, teach=False):
    # Update potentials
    for j, n in enumerate(neurons[layer - 1]):
        if n.t_rest < times[i]:
            prev_currents = currents[layer - 1][:, i]

            res_currents = multiply(prev_currents, st[layer - 1][:, i])

            # print layer, weights[layer - 1][j].shape, res_currents.shape
            dv = dot(weights[layer - 1][j], res_currents)

            n.v += dv

            if n.v > n.v_rest:
                n.v -= v_decay
    
    # Update spike trains/currents
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


def train(inputs, update_weights=None, ans=None):
    inputs = inputs.flatten()

    for layer in neurons:
        for neuron in layer:
            neuron.clear()
    
    st = [array([array(gen_st(x, len(times), mult)) for x in inputs])]
    st += [zeros((len(x), len(times))) for x in neurons]
    

    currents = [full((input_size, len(times)), i_inc)]
    currents += [zeros((len(x), len(times))) for x in neurons]

    pots = []

    for i, t in enumerate(times):
        integrate(i, 1, st, currents, bool(ans))
        integrate(i, 2, st, currents, bool(ans))
        integrate(i, 3, st, currents, bool(ans))

        pots.append([[x.v for x in layer] for layer in neurons])

        if (bool(update_weights)):
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
    
    for i in range(iters):
        for sq_i, sq_size in enumerate(square_sizes):
            for x in range(dimen - sq_size):
                for y in range(dimen - sq_size):
                    ans = zeros(len(square_sizes))
                    ans[sq_i] = 1

                    st, currents, pots = train(gen_square([x, y], sq_size, 10), update_rule, ans) 
                
    print("After: ")
    print(weights[1])
    print(weights[2])

main()

