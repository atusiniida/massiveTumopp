from vae import getModel
from loadData import genDLdata
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
help_ = "data directory"
parser.add_argument("-d", "--data", help=help_, default='data')
help_ = "result directory"
parser.add_argument("-r", "--res", help=help_,  default='result')
help_ = "image directory"
parser.add_argument("-i", "--img", help=help_,  default='image')
args = parser.parse_args()


datadir = os.path.abspath(args.data)
resdir = os.path.abspath(args.res)
imgdir = os.path.abspath(args.img)


os.makedirs(imgdir, exist_ok=True)

batch_size = 128

infiles, data =  genDLdata(datadir)
infiles = [os.path.splitext(os.path.basename(f))[0] for f in infiles]
image_size = data[0].shape[1]
data = data[:batch_size]
data = np.reshape(data, [-1, image_size, image_size, 1])
data = data.astype('float32')
data /=  np.max(data)
vae, _, _ = getModel(weights=resdir + '/model.h5')
data_decoded = vae.predict(data, batch_size=batch_size)
print(data_decoded)
for i in range(len(data)):
    input = data[i][:,:,0]
    output = data_decoded[i][:,:,0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(input, cmap="Greys")
    ax.set_title("input")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(output, cmap="Greys")
    ax.set_title("output")
    fig.tight_layout()
    fig.show()
    outfile = imgdir + '/' + infiles[i] + ".png"
    fig.savefig(outfile)
