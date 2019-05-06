from PIL import Image
import PIL
import numpy as np
from lines import gen_square

#Create input array
#input = np.asarray(PIL.Image.open('test.png'))
R = np.random.rand(10,10) * 255




#convert result into output image
result = gen_square([x, y], sq_size, dimen)

img = PIL.Image.fromarray(R.astype('uint8')).convert('L')
img.save('test2.png')