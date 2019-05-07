from numpy import *
from pylab import *
from random import randint
from math import *

from neuron import neuron
from rules import *
from spike_train import *
from lines import gen_square

from sys import argv
import pickle

iters = 3 # How many times to train

T = 25
dt = .1


i_inc = 1 # current delta for spike
i_decay = .001
i_max = .5

v_decay = .1

teach_current = 20

mult = 1 # spike value in spike train

log = False

square_sizes = [3]

dimen = 10

input_size = dimen ** 2

neurons = [
    [neuron() for x in range(input_size+1)], # Input Layer
    [neuron() for x in range(1 + sum([(dimen-sq_size+1) ** 2 for sq_size in square_sizes]))], # Bounding Box Layer
    [neuron() for x in range(len(square_sizes) + dimen**2)] # Output OR Gate
]

def gen_weights(input_size, neurons):
    weights = [full((input_size+1, len(neurons[0])), 0)]


    for i in range(input_size):
        weights[0][i][i] = 45

    for i in range(1, len(neurons)):
        # weights.append(np.random.uniform(0, 1, (len(neurons[i]), len(neurons[i - 1]))))
        weights.append(full((len(neurons[i]), len(neurons[i - 1])), 1))

        '''for k in range(len(neurons[i-1])):
            weights[-1][k][len(neurons[i - 1])-1] = 0.00001
            print(weights[-1][len(neurons[i])-1][k])'''
        for j in range(len(neurons[i])):
            weights[i][j][-1] = .00001

    # print("YOYOYO")
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

            # tes = weights[layer - 1][j][j]


            # print(layer)
            # print(weights[layer - 1].shape)
            # print(res_currents.shape)
            dv = dot(weights[layer - 1][j], res_currents)

            '''if layer == 2:
                temp = st[layer - 1][:, i]
                print("RESULT")
                print(weights[layer - 1][j][-1])
                print(j, dv)'''


            n.v += dv

            if n.v > n.v_rest:
                n.v -= v_decay

    # Update spike trains/currents
    for j, n in enumerate(neurons[layer -  1]):
        res = n.is_spike()
        currents[layer][j][i] =  currents[layer][j][i - 1]

        if res:
            # print ("Spiked ", layer, j, i)
            currents[layer][j][i] += i_inc

            n.t_rest = times[i] + n.t_delay
            n.v = n.v_rest

            st[layer][j][i] = 1

        if currents[layer][j][i] > 0:
            currents[layer][j][i] -= i_decay # maybe allow neg currents


def update_ojas(layer, rates, neurons):
    for j, n in enumerate(neurons[layer]):
        for k, f in enumerate(neurons[layer - 1]):

            if k == (len(neurons[layer - 1]) - 1):
                # print("vro")
                continue
            elif j == (len(neurons[layer]) - 1):
                continue
            else:
                pre = rates[layer-1][k]
                post = rates[layer][j]

                updates = ojas(post, pre, weights[layer][j][k])

                if layer == 1:
                    print(j, updates, pre, post)

                # if(f == 1):
                #     if(s == 0):
                #         pass
                        # print(updates, f, s, j, k)




                weights[layer][j][k] += updates


def train(inputs, update_weights=None, ans=None, ans_two=None, ans_three=None):
    print("RIGHT ANSWER")
    print(ans, ans_two, ans_three)

    inputs = inputs.flatten()

    for layer in neurons:
        for neuron in layer:
            neuron.clear()

    st = [array([array(gen_st(x, len(times), mult)) for x in inputs])]

    st += [zeros((len(x), len(times))) for x in neurons]
    st[0] = np.append(st[0], [[1]*len(times)], axis=0)
    st[1][-1] = array([[1]*len(times)])
    st[2][-1] = array([[1]*len(times)])

    if ans is not None:
        weights[1][ans][-1] = 3
        weights[2][ans_two][-1] = 3
        weights[2][dimen**2 + ans_three][-1] = 3
        
    currents = [full((input_size+1, len(times)), i_inc)]
    currents += [zeros((len(x), len(times))) for x in neurons]

    currents[1][-1] = array([[teach_current] * len(times)])
    currents[2][-1] = array([[teach_current] * len(times)])

    # currents[2] = np.append(currents[2], [[teach_current] * len(times)], axis=0)
    # print(currents[0])

    pots = []

    # print(currents[0].shape)
    # print(currents[1].shape)
    # print(currents[2].shape)
    # print(currents[3].shape)
    # print(len(neurons[0]))
    # print(len(neurons[1]))
    # print(len(neurons[2]))

    for i, t in enumerate(times):
        integrate(i, 1, st, currents, bool(ans))

        integrate(i, 2, st, currents, bool(ans))

        integrate(i, 3, st, currents, bool(ans))

        pots.append([[x.v for x in layer] for layer in neurons])

    rates = []
    for l, st_layer in enumerate(st):
        if l == 0:
            continue
        rates.append([sum(i)/T for i in st[l]])

    if (bool(update_weights)):
        update_weights(1, rates, neurons)
        update_weights(2, rates, neurons)

    print("RES FIRST", [sum(x) for x in st[1]])
    print("RES SECOND", [sum(x) for x in st[2]])
    print("RES THIRD", [sum(x) for x in st[3]])

    if ans is not None:
        weights[1][ans][-1] = 0
        weights[2][ans_two][-1] = 0
        weights[2][dimen ** 2 + ans_three][-1] = 0

    print(ans_two)

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
    with open("squares.txt", 'rb') as f:
        examples = pickle.load(f)
        matrices = examples[0]
        answers = examples[1]
        if len(argv) == 2:
            print("Start Weights: ")
            print(weights[1])
            print(weights[2])

            for i in range(4):
                print(i)
                for j in range(1):
                    k = 0

                    print(matrices[j])
                    print(answers[j])
                    st, currents, pots = train(matrices[k], update_ojas, answers[k][0], answers[k][1], answers[k][2])
                    last_layer = st[-1]

                    sums = [sum(x) for x in last_layer]

                    print(sums)
                    print(weights[1])
                    print(weights[2])
                    print()

                    # return

            print("After: ")
            print(weights[1])
            print(weights[2])
            np.save("weights1", weights[1])
            np.save("weights2", weights[2])

        print("\n")

        print("Running: ")
        weights[1] = np.load("weights1.npy")
        weights[2] = np.load("weights2.npy")

    with open('test_squares.txt', 'rb+') as f:
        test_squares = pickle.load(f)
        matrices = test_squares[0]
        for matrix in matrices:
            print(matrix)
            st, currents, pots = train(matrix, None)
            last_layer = st[-1]
            sums = [sum(x) for x in last_layer]
            print(sums)
            print("\n")

if __name__ == "__main__":
    main()

