# coding: utf-8
__author__ = 'zq'


# from __future__ import print_function
from getFeature import get_feature
import lasagne
import theano
import theano.tensor as T
import itertools
import numpy as np
import cPickle
import gzip
import glob
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer



# NUM_EPOCHS = 300
# # Osize = 216
Osize = 240
BATCH_SIZE = 10
Input_Shape = None, 3, Osize, Osize
Output_Shape = None, 1, Osize/3, Osize/3
input_height = Input_Shape[2]
input_width = Input_Shape[3]
output_dim = Output_Shape[1] * Output_Shape[2] * Output_Shape[3]

saveParamName= 'GEM.pkl.gz'

def get_PicPathNameList(top, TRAIN_set_num=0, VALID_set_num=0, TEST_set_num=1000):
    print("start getting the name list of images")
    if top.__class__ == "123".__class__:
        inputNameList = glob.glob(top+"/*.jpg")
        targetNameList = [s.replace('.jpg', '.png') for s in inputNameList]
    else:
        inputNameList = glob.glob(top[0]+"/*.png")
        inputNameList.extend(glob.glob(top[0]+"/*.jpg"))
        targetNameList = glob.glob(top[1]+"/*.*")
    print("get "+str(len(inputNameList))+" images")
    return dict(
        X_train=inputNameList[0:TRAIN_set_num],
        y_train=targetNameList[0:TRAIN_set_num],
        X_valid=inputNameList[TRAIN_set_num:TRAIN_set_num + VALID_set_num],
        y_valid=targetNameList[TRAIN_set_num:TRAIN_set_num + VALID_set_num],
        X_test=inputNameList[TRAIN_set_num + VALID_set_num::],
        y_test=targetNameList[TRAIN_set_num + VALID_set_num::],
        )


def main():

    X_batch = T.tensor4('x')
    print("Building model and compiling functions...")
    l_in = lasagne.layers.InputLayer(
        shape=(None, 3, input_height, input_width),
        input_var=X_batch,
        )
    l_conv1 = ConvLayer(
        l_in,
        num_filters=32,
        filter_size=(13, 13),
        pad=6,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        # border_mode='same',
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, (4, 4))
    l_conv2_1 = ConvLayer(
        l_pool1,
        num_filters=64,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_conv2_2 = ConvLayer(
        l_conv2_1,
        num_filters=96,
        filter_size=(5, 5),
        pad=2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_conv2_3 = ConvLayer(
        l_conv2_2,
        num_filters=96,
        filter_size=(9, 9),
        pad=4,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_3, (3, 3))
    l_conv3_1 = ConvLayer(
        l_pool2,
        num_filters=128,
        filter_size=(3, 3),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_conv3_2 = ConvLayer(
        l_conv3_1,
        num_filters=128,
        filter_size=(7, 7),
        pad='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )
    l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3_2, (2, 2))
    l_conv4 = ConvLayer(
        l_pool3,
        num_filters=160,
        filter_size=(3, 3),
        pad='same',
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
        flip_filters=True,
        )

    l_net1 = l_conv4
    l_up = lasagne.layers.Upscale2DLayer(l_net1, (2, 2))
    l_deconv1 = ConvLayer(l_up, 128, (9, 9), pad='same', nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         flip_filters=True
                         )
    l_deconv2 = ConvLayer(l_deconv1, 96, (5, 5), pad='same', nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         flip_filters=True
                         )
    l_deconv3 = ConvLayer(l_deconv2, 96, (3, 3), pad='same', nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         flip_filters=True
                         )
    l_up = lasagne.layers.Upscale2DLayer(l_deconv3, (2, 2))
    l_deconv4 = ConvLayer(l_up, 96, (5, 5), pad='same', nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         flip_filters=True
                         )
    l_deconv5 = ConvLayer(l_deconv4, 96, (3, 3), pad='same', nonlinearity=lasagne.nonlinearities.leaky_rectify,
                         flip_filters=True
                         )
    l_up = lasagne.layers.Upscale2DLayer(l_deconv5, (2, 2))
    l_deconv6 = ConvLayer(l_up, 32, (5, 5), pad='same', nonlinearity=lasagne.nonlinearities.rectify,
                         flip_filters=True
                         )
    l_deconv7 = ConvLayer(l_deconv6, 32, (3, 3), pad='same', nonlinearity=lasagne.nonlinearities.rectify,
                         flip_filters=True
                         )
    l_deconv = ConvLayer(l_deconv7, 1, (1, 1), nonlinearity=lasagne.nonlinearities.sigmoid)
    l_out = lasagne.layers.flatten(l_deconv)

    output_layer = l_out

    feature = lasagne.layers.get_output(output_layer)
    f = gzip.open(saveParamName, 'rb')
    params = cPickle.load(f)
    f.close()
    lasagne.layers.set_all_param_values(output_layer, params)
    data = get_PicPathNameList('./image')
    get_feature(feature, X_batch, data)
if __name__ == '__main__':
    main()
