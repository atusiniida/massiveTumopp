import csv
from os import listdir
import numpy as np
import re
import os
import glob
from PIL import Image

size = 256
def loadSingleFile(fileName):

    with open(fileName, "r") as f:
        a = np.array(list(csv.reader(f, delimiter='\t')), dtype=np.float32)
        i = Image.fromarray(np.float32(a))
        a = np.asarray(i.resize((size, size)))
        return a


def genDLdata(dataPath):

    cachePath = dataPath+"/cache.npy"
    files = glob.glob(dataPath + "/*.tab")
    if os.path.exists(cachePath):
        data = np.load(cachePath)
        return (files, data)

    data = [loadSingleFile(f) for f in files]
    np.save(cachePath, data)

    return (files, data)


if __name__ == "__main__":
    homedir = os.path.abspath(os.path.dirname(__file__))
    homedir = re.sub("/python", "", homedir)
    dataPath = homedir + "/data"
    (files, data) = genDLdata(dataPath)
    print("@ Loading from cache okay!")
