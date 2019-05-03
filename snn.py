from numpy import *
from pylab import *
from random import randint
from math import *

from neuron import neuron
from rules import *
from spike_train import *
from lines import gen_square

from sys import argv

iters = 3 # How many times to train

T = 25
dt = .1


i_inc = .5 # current delta for spike
i_decay = .001
i_max = .5

v_decay = .1

teach_current = 20

mult = 1 # spike value in spike train

log = False

square_sizes = [8]
test_square_sizes = [3,5,8]

def_size = 4 # default square size
dimen = 10

input_size = dimen ** 2

neurons = [
    [neuron() for x in range(input_size+1)], # Input Layer
    [neuron() for x in range(1 + sum([(dimen-sq_size+1) ** 2 for sq_size in square_sizes]))], # Bounding Box Layer
    [neuron() for x in range(len(square_sizes) + dimen**2)] # Output OR Gate
]

def gen_weights(input_size, neurons):
    weights = []
    weights.append(np.random.randn(input_size+1, len(neurons[0])))
    for i in range(1, len(neurons)):
        weights.append(np.random.randn(len(neurons[i]), len(neurons[i - 1])))
    print(weights[0].shape)
    print(weights[1].shape)
    print(weights[2].shape)
    return weights

weights = gen_weights(dimen ** 2, neurons)
# weights[1] = np.append(weights[1], full((len(neurons[1]), 1), 1), axis=1)

times = arange(0, T + dt, dt)

# integrate even the first layer
def integrate(i, layer, st, currents, teach=False):
    # Update potentials
    for j, n in enumerate(neurons[layer - 1]):
        if n.t_rest < times[i]:
            # print(currents[layer-1])
            prev_currents = currents[layer - 1][:, i]

            # print(prev_currents.shape)
            res_currents = multiply(prev_currents, st[layer - 1][:, i])

            # print(layer)
            # print(weights[layer - 1].shape)
            # print(res_currents.shape)
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


def train(inputs, update_weights=None, ans=None, ans_two=None, ans_three=None):
    if update_weights is None:
        weights[1] = np.load("weights1.npy")
        weights[2] = np.load("weights2.npy")
    inputs = inputs.flatten()

    for layer in neurons:
        for neuron in layer:
            neuron.clear()

    st = [array([array(gen_st(x, len(times), mult)) for x in inputs])]
    st += [zeros((len(x), len(times))) for x in neurons]
    st[0] = np.append(st[0], [[1]*len(times)], axis=0)
    # st[2] = np.append(st[2], [[1] * len(times)], axis=0)

    if ans is not None:
        weights[1][ans][-1] = 1
    else:
        weights[1][ans][-1] = 0

    if ans_two is not None:
        weights[2][ans_two][-1] = 1
        weights[2][dimen**2 + ans_three][-1] = 1
    else:
        weights[2][ans_two][-1] = 0
        weights[2][dimen ** 2 + ans_three][-1] = 0

    currents = [full((input_size+1, len(times)), i_inc)]
    currents += [zeros((len(x), len(times))) for x in neurons]

    # currents[1] = np.append(currents[1], [[teach_current] * len(times)], axis=0)
    # currents[2] = np.append(currents[2], [[teach_current] * len(times)], axis=0)
    # print(currents[0])

    pots = []

    print(currents[0].shape)
    print(currents[1].shape)
    print(currents[2].shape)
    print(currents[3].shape)
    # print(len(neurons[0]))
    # print(len(neurons[1]))
    # print(len(neurons[2]))

    for i, t in enumerate(times):
        integrate(i, 1, st, currents, bool(ans))
        integrate(i, 2, st, currents, bool(ans))
        integrate(i, 3, st, currents, bool(ans))

        pots.append([[x.v for x in layer] for layer in neurons])

        if (bool(update_weights)):
            update_weights(i, 2, st, neurons)
            update_weights(i, 3, st, neurons)

    weights[1][ans][-1] = 0

    weights[2][ans_two][-1] = 0
    weights[2][dimen ** 2 + ans_three][-1] = 0

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
    if len(argv) == 2:
        update_rule = update_ojas
    else:
        update_rule = None

    print("Start Weights: ")
    print(weights[1])
    print(weights[2])

    for i in range(iters):
        print(i)
        cumul = 0
        for sq_i, sq_size in enumerate(square_sizes):
            track = 0
            for x in range(dimen - sq_size + 1):
                for y in range(dimen - sq_size + 1):
                    print("at (" + str(x) + ", " + str(y) + ")")
                    ans = cumul + track
                    st, currents, pots = train(gen_square([x, y], sq_size, 10), update_rule, ans, dimen*x+y, sq_i)
                    last_layer = st[-1]
                    sums = [sum(x) for x in last_layer]
                    print(sums)
                    track += 1
                    print(weights)
            cumul += (dimen-sq_size+1)**2

    print("After: ")
    print(weights[1])
    print(weights[2])
    np.save("weights1", weights[1])
    np.save("weights2", weights[2])

main()

