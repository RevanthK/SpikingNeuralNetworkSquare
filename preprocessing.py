from PIL import Image
import PIL
import numpy as np
from lines import gen_square
from snn import train

def parseOutputLayer(sums, dimen):
    max = 0
    x = -1, y = -1
    sq_size = 0
    temp_size = 3

    counter = 0
    for i in sums and (dimen - 2) >= 1:
        if counter < (dimen - 2) ** 2:
            if i > max:
                max = i
                sq_size = temp_size
                x = temp_size + counter % (dimen - 2)
                y = temp_size + counter / dimen - 2
        else:
            dimen -= 1
            counter = 0
            temp_size += 1

    return x, y, sq_size



#Create input array
input = np.asarray(PIL.Image.open('test.png'))
#input = np.random.rand(10,10) * 255
dimen = input.shape[0]

st, currents, pots = train(input, None)
last_layer = st[-1]
sums = [sum(x) for x in last_layer]

x, y, sq_size = parseOutputLayer(sums, dimen)

#convert result into output image
result = gen_square([x, y], sq_size, dimen)

img = PIL.Image.fromarray(input.astype('uint8')).convert('L')
img.save('test2.png')