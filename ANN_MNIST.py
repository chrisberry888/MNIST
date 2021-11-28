import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.neural_network import MLPClassifier
import random

random.seed(11)

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

train_images = [x.flatten() for x in train_images]
test_images = [x.flatten() for x in test_images]

#plt.imshow(train_images[4], cmap=plt.cm.binary)
#plt.draw()
print(len(train_images[0]))
#print(train_labels[:10])
#print(train_images[4])
print(1)
clf = MLPClassifier(hidden_layer_sizes=())
clf1 = MLPClassifier(hidden_layer_sizes=(100,))
print(2)
clf.fit(train_images, train_labels)
print(3)
print(clf.score(test_images,test_labels))
print(4)
clf1.fit(train_images, train_labels)
print(5)
print(clf1.score(test_images,test_labels))
print(6)