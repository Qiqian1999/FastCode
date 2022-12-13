import sys
import caffe
from PIL import Image
import numpy as np
import time

# nimg = np.random.rand(1, 3, 224, 224) # ('execution time (s): ', 1.862389087677002)
# nimg = np.random.rand(16, 3, 224, 224) # ('execution time (s): ', 25.17376685142517)
nimg = np.random.rand(32, 3, 224, 224) # ('execution time (s): ', 49.40081191062927)

caffe.set_mode_cpu()
net = caffe.Net("deploy.prototxt", "VGG_ILSVRC_16_layers.caffemodel", caffe.TEST)
# net.blobs['data'].reshape(1, 3, 224, 224)
# net.blobs['data'].reshape(16, 3, 224, 224)
net.blobs['data'].reshape(32, 3, 224, 224)
net.blobs['data'].data[...] = nimg

print('warming up...')
for i in range(10):
    out = net.forward()

print('fowarding...')
t = time.time()
out = net.forward()
elapsed = time.time() - t
print("execution time (s): ", elapsed)
