# coding: utf-8
#
# from __future__ import print_function

# import lasagne
import theano
import theano.tensor as T
import numpy as np
import math
from PIL import Image
from PIL import ImageOps
import random
import thread
import copy

MaxPic = 256

def read_pics(inputNamebatch, batchsize, crop, mirror, flip, rotate):
    inputsbatch = []
    for i in range(batchsize):
        try:
            img = Image.open(inputNamebatch[i])
            img = img.resize((216, 216))
            # label = Image.open(targetNamebatch[i])
        except:
            img = None
            # label = None
        img.getdata()
        image = img.split()
        if len(image)==1:
            img = np.asarray((image[0]), dtype='float32')/MaxPic
            img = np.asarray((img, img, img), dtype='float32')
        else:
            img1 = np.asarray((image[0]), dtype='float32')/MaxPic
            img2 = np.asarray((image[1]), dtype='float32')/MaxPic
            img3 = np.asarray((image[2]), dtype='float32')/MaxPic
            img = np.asarray((img1, img2, img3), dtype='float32')
        # img = np.asarray(img.split(), dtype='float32')/MaxPic
        inputsbatch.extend([[]])
        inputsbatch[i] = img
    return np.asarray(inputsbatch, dtype='float32')

def iterate_minibatches_feature(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        inputsbatch = read_pics(inputs[start_idx:start_idx + batchsize],batchsize,False,False,False,False)
        yield inputsbatch


def get_feature(test_prediction, X_batch, dataset, batch_size=1):
    feature_fn = theano.function([X_batch], [test_prediction])
    save_list = range(0, len(dataset['X_test']) - batch_size + 1, batch_size)
    batchnum=0
    for batch in iterate_minibatches_feature(dataset['X_test'], batch_size):
        inputs = batch
        [batch_ypred]= feature_fn(inputs)
        save_listbatch = dataset['X_test'][save_list[batchnum]: save_list[batchnum] + batch_size]
        batchnum+=1
        try:
            for i in range(len(batch_ypred)):
                for j in range(len(batch_ypred[i])):
                    map=batch_ypred[i,j]
                    map=(map-map.min())/(map.max()-map.min())
                    Image.fromarray((map*255).astype(np.uint8)).convert("L").save(save_listbatch[i] + ".fe" + str(j) + ".bmp", "bmp")
        except:
            size = math.sqrt(len(batch_ypred[0]))
            map=batch_ypred[0]
            map=(map-map.min())/(map.max()-map.min())
            Image.fromarray((np.resize(np.asarray(map), (size, size))*255).astype(np.uint8)).convert("L").save(save_listbatch[0][0:-4] + "_GEM" + ".bmp", "bmp")