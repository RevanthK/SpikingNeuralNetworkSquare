from numpy import *
from snn import gen_weights, weights

weights[1] = load("weights1.npy")
weights[2] = load("weights2.npy")

print weights[1]