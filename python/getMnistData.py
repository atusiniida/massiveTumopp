import os
import keras
from keras.datasets import mnist
import numpy as np

outdir = "mnist"
os.makedirs(outdir, exist_ok=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(len(x_train)):
    np.savetxt(outdir + "/" + str(i) +".tab", x_train[i], delimiter = "\t", fmt = "%d")
