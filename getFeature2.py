# coding: utf-8
# 5-20-1
# from __future__ import print_function

# import lasagne
import theano
import theano.tensor as T
import time
import numpy as np
import math
from PIL import Image
from PIL import ImageOps
import random
import thread
import copy

MaxPic = 256

def read_pics(inputNamebatch, input2Namebatch, batchsize, crop, mirror, flip, rotate):
    inputsbatch = []
    inputsbatch2 = []
    for i in range(batchsize):
        img = Image.open(inputNamebatch[i])
        img = img.resize((int(img.size[0]/2)*2, int(img.size[1]/2)*2 ))
        img_2 = Image.open(input2Namebatch[i])
        img_2 = img_2.resize((int(img.size[0]/2), int(img.size[1]/2)))
            # label = Image.open(targetNamebatch[i])
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


        img_2.getdata()
        img_2 = np.asarray(img_2, dtype='float32')/MaxPic

        # img = np.asarray(img.split(), dtype='float32')/MaxPic
        inputsbatch.extend([[]])
        inputsbatch2.extend([[[]]])
        inputsbatch[i] = img
        inputsbatch2[i][0] = img_2
    return np.asarray(inputsbatch, dtype='float32'), np.asarray(inputsbatch2, dtype='float32')

def iterate_minibatches_feature(inputs, batchsize):
    if len(inputs)==2:
        inputs2 = inputs[1]
        inputs = inputs[0]
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        inputsbatch, inputs2batch = read_pics(inputs[start_idx:start_idx + batchsize], inputs2[start_idx:start_idx + batchsize], batchsize,False,False,False,False)
        yield inputsbatch, inputs2batch


def get_feature2(test_prediction, X_batch, X2_batch, dataset, batch_size=1):
    try:
        feature_fn = theano.function([X_batch, X2_batch], [test_prediction])
    except:
        feature_fn = theano.function([X_batch], [test_prediction])
    save_list = range(0, len(dataset['X_test'][0]) - batch_size + 1, batch_size)
    batchnum=0
    for batch in iterate_minibatches_feature(dataset['X_test'], batch_size):
        inputs, inputs2 = batch
        try:
            [batch_ypred]= feature_fn(inputs,inputs2)
        except:
            [batch_ypred]= feature_fn(inputs)
        save_listbatch = dataset['X_test'][0][save_list[batchnum]: save_list[batchnum] + batch_size]
        batchnum+=1
        for i in range(len(batch_ypred)):
            for j in range(len(batch_ypred[i])):
                map=batch_ypred[i,j]
                map=(map-map.min())/(map.max()-map.min())
                Image.fromarray((map*255).astype(np.uint8)).convert("L").save(save_listbatch[i][0:-4] + "_GERM" + ".bmp", "bmp")