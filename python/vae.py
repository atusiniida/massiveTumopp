from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import *
from keras.models import *
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import csv
import sys

from loadData import genDLdata


def plotLog(infile,outfile):
    csv_file = open(infile, "r")
    f = csv.reader(csv_file)
    header = next(f)
    epochs = []
    loss = []
    val_loss = []
    acc = []
    val_acc = []

    # print(header)
    for row in f:
        epochs.append(int(row[0])+1)
        loss.append(float(row[1]))
        val_loss.append(float(row[1]))
        # print(row)
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, loss, marker='.', label='loss')
    plt.plot(epochs, val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(ymin=0.0)

    plt.tight_layout()
    # plt.show()
    plt.savefig(outfile)

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def getModel(weights="",  outdir=""):
    # parameters
    image_size = 256
    input_shape = (image_size, image_size, 1)
    latent_dim = 16

    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for n in [32,64,128,128,128]:
        x = Conv2D(n, kernel_size=3, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2)(x)

    shape = K.int_shape(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    if outdir:
        plot_model(encoder, to_file= outdir + '/vae_cnn_encoder.png', show_shapes=True)

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape(shape[1:])(x)
    for n in [128,128,64,32,1]:
        x = UpSampling2D(size=2)(x)
        x = Conv2D(n, kernel_size=3, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
    x = Conv2D(1, kernel_size=3, padding='same')(x)
    outputs = Activation('sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    if outdir:
        plot_model(decoder, to_file= outdir + '/vae_cnn_decoder.png', show_shapes=True)


    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    #reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    if outdir:
        plot_model(vae, to_file = outdir + '/vae_cnn.png', show_shapes=True)
    if weights:
        vae.load_weights(weights)
    return (vae, encoder, decoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "data directory"
    parser.add_argument("-d", "--data", help=help_, default='data')
    help_ = "result directory"
    parser.add_argument("-r", "--res", help=help_,  default='result')
    args = parser.parse_args()


    indir = os.path.abspath(args.data)
    outdir = os.path.abspath(args.res)

    os.makedirs(outdir, exist_ok=True)

    batch_size = 128
    epochs = 100

    _, data =  genDLdata(indir)
    image_size = data[0].shape[1]
    data = np.reshape(data, [-1, image_size, image_size, 1])
    data = data.astype('float32')
    data /=  np.max(data)

    vae, _, _= getModel(outdir=outdir)

    # train the autoencoder
    csv_logger = CSVLogger(outdir + '/log.cvs')
    es = EarlyStopping(monitor='val_loss', patience=2)
    if os.path.exists(outdir + '/model.h5'):
        print("A model file exists!", file=sys.stderr)
        exit()
    vae.fit(data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[es, csv_logger])
            # callbacks=[csv_logger])
    vae.save_weights(outdir + '/model.h5')
    plotLog(outdir + '/log.cvs', outdir + '/log.png')
