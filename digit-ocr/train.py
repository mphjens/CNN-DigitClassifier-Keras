import numpy as np
import os
from os import listdir
from os.path import isfile, join
import struct
import random

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils

from scipy import ndimage

import image


MODEL_FILENAME = 'digit.pkl'
WEIGHTS_FILENAME = 'weights1.h5'

def getFilenames(path=".", shuffled=False):
    fileNames = [f for f in listdir(path) if isfile(join(path, f))]

    if shuffled:
        random.shuffle(fileNames)

    return fileNames


def read_dset(filenames, startIndex = 0, endIndex = None):
    imgs = []
    labels = []
    data = []
    for name in filenames[startIndex:endIndex]:
        img = ndimage.imread("/home/jens/dev/Python-Snippets/digits/" +name)
        components = name.split("_")
        label = float(components[0])
        # imgs.append(img)
        # labels.append(label)
        data.append((label,img))

    return data


#Convert the read images to a normalized numpy array with the right shape
def toNpdata(data, maxItems=30000):
    datalist = [t for t in data]
    m = maxItems
    n = 28
    X = np.zeros((m, 1, n, n))
    Y = np.zeros(m)
    for i, (label, image) in enumerate(datalist[:m]):
        X[i,0, :] = image.reshape(28,28)
        Y[i] = label

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx].astype(np.float)/255, Y[idx]

def get_CNN():
    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dropout(0.25)) #Dropout to reduce overfitting
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25)) #Dropout to reduce overfitting
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_CNN():
    fnames = getFilenames("/home/jens/dev/Python-Snippets/digits", shuffled = True)
    X_train, y_train = toNpdata(read_dset(fnames, startIndex=0, endIndex=1800), maxItems=1800)
    X_test, y_test = toNpdata(read_dset(fnames, startIndex=1800, endIndex=2000), maxItems=200)

    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = get_CNN()

    model.fit(X_train, Y_train,
              batch_size=16, nb_epoch=5, validation_data=(X_test,Y_test))


    model.save_weights("CNN_"+WEIGHTS_FILENAME)
    score = model.evaluate(X_test, Y_test,
                              verbose=1)
    print ('Test score: ' + str(score))

def get_model():
    model = get_CNN()
    model.load_weights("CNN_"+WEIGHTS_FILENAME)
    return model

if __name__ == '__main__':
    train_CNN()
