__author__ = 'zq'
#coding: utf-8

# from __future__ import print_function
from getFeature2 import get_feature2
import lasagne
import theano
import theano.tensor as T
import time
import itertools
import numpy as np
import cPickle
import gzip
import glob
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer


BATCH_SIZE = 1
saveParamName = 'GERM.pkl.gz'
LEARNING_RATE = 0.005
MOMENTUM = 0.9
def get_PicPathNameList2(top, TRAIN_set_num=8000, VALID_set_num=1000, TEST_set_num=1000):
    print("start getting the name list of images")
    inputNameList = glob.glob(top+"/*.jpg")
    targetNameList = [s.replace('.jpg', '.png') for s in inputNameList]
    input2NameList = [s.replace('.jpg', "_GEM.bmp") for s in inputNameList]
    print("get "+str(len(inputNameList))+" images")
    if input2NameList == []:
        return dict(
            X_train=inputNameList[0:TRAIN_set_num],
            y_train=targetNameList[0:TRAIN_set_num],
            X_valid=inputNameList[TRAIN_set_num:TRAIN_set_num + VALID_set_num],
            y_valid=targetNameList[TRAIN_set_num:TRAIN_set_num + VALID_set_num],
            X_test=inputNameList[TRAIN_set_num + VALID_set_num::],
            y_test=targetNameList[TRAIN_set_num + VALID_set_num::],
        )
    else:
        X2_train=input2NameList[0:TRAIN_set_num]
        X2_valid=input2NameList[TRAIN_set_num:TRAIN_set_num + VALID_set_num]
        X2_test=input2NameList[TRAIN_set_num + VALID_set_num::]
        return dict(
            X_train=(inputNameList[0:TRAIN_set_num], X2_train),
            y_train=targetNameList[0:TRAIN_set_num],
            X_valid=(inputNameList[TRAIN_set_num:TRAIN_set_num + VALID_set_num], X2_valid),
            y_valid=targetNameList[TRAIN_set_num:TRAIN_set_num + VALID_set_num],
            X_test=(inputNameList[TRAIN_set_num + VALID_set_num::], X2_test),
            y_test=targetNameList[TRAIN_set_num + VALID_set_num::],
            )

def main():

    X_batch = T.tensor4('x')
    X2_batch = T.tensor4('x2')
    y_batch = T.tensor4('y')
    print("Building model and compiling functions...")
    l_input2 = lasagne.layers.InputLayer(
        shape=(None, 1, None, None),
        input_var=X2_batch
        )

    l_in = lasagne.layers.InputLayer(
        shape=(None, 3, None, None),
        input_var=X_batch,
        )
    l_conv1 = Conv2DLayer(
        l_in,
        num_filters=32,
        filter_size=(13, 13),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, (2, 2))

    l_conv2_1 = Conv2DLayer(
        l_pool1,
        num_filters=96,
        filter_size=(3, 3),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_conv2_2 = Conv2DLayer(
        l_conv2_1,
        num_filters=95,
        filter_size=(5, 5),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_merge = lasagne.layers.ConcatLayer((l_conv2_2, l_input2))

    l_conv2_3 = Conv2DLayer(
        l_merge,
        num_filters=128,
        filter_size=(5, 5),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_conv2_4 = Conv2DLayer(
        l_conv2_3,
        num_filters=192,
        filter_size=(3, 3),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_up = lasagne.layers.Upscale2DLayer(l_conv2_4, (2, 2))
    l_conv3_1 = Conv2DLayer(
        l_up,
        num_filters=128,
        filter_size=(3, 3),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_conv3_2 = Conv2DLayer(
        l_conv3_1,
        num_filters=96,
        filter_size=(3, 3),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )

    l_conv = Conv2DLayer(l_conv3_2, 1, (1, 1), nonlinearity=lasagne.nonlinearities.sigmoid)


    output_layer = l_conv

    feature = lasagne.layers.get_output(l_conv)
    f = gzip.open(saveParamName, 'rb')
    params = cPickle.load(f)
    f.close()
    lasagne.layers.set_all_param_values(output_layer, params)
    data = get_PicPathNameList2('./image',TRAIN_set_num=0,VALID_set_num=0)
    get_feature2(feature, X_batch, X2_batch, data)

if __name__ == '__main__':
    main()
