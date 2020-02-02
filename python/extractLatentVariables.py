from vae import getModel
from loadData import genDLdata
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
help_ = "data directory"
parser.add_argument("-d", "--data", help=help_, default='data')
help_ = "result directory"
parser.add_argument("-r", "--res", help=help_,  default='result')
args = parser.parse_args()

indir = os.path.abspath(args.data)
outdir = os.path.abspath(args.res)

batch_size = 128

infiles, data =  genDLdata(indir)
infiles = [os.path.splitext(os.path.basename(f))[0] for f in infiles]
image_size = data[0].shape[1]
data = np.reshape(data, [-1, image_size, image_size, 1])
data = data.astype('float32') / 255

_, encoder, _ = getModel(weights=outdir + '/model.h5')
z_mean, _, _ = encoder.predict(data, batch_size=batch_size)
f = open(outdir + '/latent.tab', "w")
f.write("\t" + "\t".join([ "z[" + str(j) + "]" for j in range(z_mean.shape[1])]) + "\n")
for i in range(len(infiles)):
    f.write(infiles[i] + "\t" + "\t".join([ str(z_mean[i,j]) for j in range(z_mean.shape[1])]) + "\n")
f.close()
