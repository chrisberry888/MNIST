import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import os, sys

plt.ion()
path = os.path.abspath(os.path.dirname(sys.argv[0]))

train_image_file = str(path) + "/data/train-images.idx3-ubyte"
train_label_file = str(path) + "/data/train-labels.idx1-ubyte"
test_image_file = str(path) + "/data/t10k-images.idx3-ubyte"
test_label_file = str(path) + "/data/t10k-labels.idx1-ubyte"

train_images = idx2numpy.convert_from_file(train_image_file)
train_labels = idx2numpy.convert_from_file(train_label_file)
test_images = idx2numpy.convert_from_file(test_image_file)
test_labels = idx2numpy.convert_from_file(test_label_file)

plt.imshow(test_images[4], cmap=plt.cm.binary)
plt.draw()

print(test_labels[:10])