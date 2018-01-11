# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:51:14 2017

@author: shiro
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:07:23 2017

@author: shiro
"""

import tensorflow as tf
import os
from scipy import signal
import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from scipy.io import wavfile
from pydub import AudioSegment
import csv
import keras 
from keras import backend as K
## - Keras - ##
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import SeparableConv2D, Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten, Reshape, Activation, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import SeparableConvolution2D
from keras.models import model_from_json
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import pickle
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

keras.metrics.f1 = f1

def relu6(x):
    return K.relu(x, max_value=6) 
    
def convert_pred(pred):
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silence', 'unknown']
    return train_words[pred]

def get_raw(paths, nsamples=16000, path=''):
    '''
    Given list of paths, return raw data
    nsample = number of samples per second.
    '''
    T = len(paths)
    #print('Size : ', T)
    # read the wav files
    wavs = [wavfile.read(path+x)[1] for x in paths]
    
    # zero pad the shorter samples and cut off the long ones.
    data = [] 
    
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
        else:
            d = wav[0:nsamples]
        
        data.append(d)


       
    return np.asarray(data, dtype=np.int16).reshape(T, -1, 1) #format conv 1D

    
def write_result(name_model, file_test, path_test, name, batch_size=6, n_samples=16000):
    ## load model
    #model = model_from_json(name_model+'.json')
    print('load json')
    model = load_model(name_model)#load_model(name_model+'-'+epoch+'.hdf5')
    f = open(file_test, 'r')
    
    file_name = [line.split()[0] for line in f]
    print(file_name[0:10])
    f.close()
    ite = int(len(file_name)/batch_size)
    print('number of iteration :', ite)
    
    Y = np.zeros( (len(file_name), 30) , dtype=np.float32)
    ## write file
    with open(name, 'w', newline='') as csvfile:
        c = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(["fname","label"])
        for cpt in range(0,ite):
        ## load data to test
            X = get_raw(file_name[cpt*batch_size:cpt*batch_size+batch_size], nsamples=n_samples, path=path_test)
            ## prediction
            #print(X.shape)
            proba = model.predict_on_batch(X)
            Y[cpt*batch_size:cpt*batch_size+batch_size, :] = proba
            pred = np.argmax(proba, axis=1)
            pred[pred>=11] = 11
            for i,p in enumerate(pred):
                c.writerow([file_name[cpt*batch_size+i], convert_pred(p)])   
    pickle.dump(Y, open(name_model.replace('.hdf5', '.pkl'), 'wb'))
    
    
def write_resultv2(name_model, epoch, file_test, path_test, name, batch_size=6,n_samples=16000):
    ## load model
    #model = model_from_json(name_model+'.json')
    print('load json')
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(name_model+'.h5')#load_model(name_model+'-'+epoch+'.hdf5')
    f = open(file_test, 'r')
    
    file_name = [line.split()[0] for line in f]
    print(file_name[0:10])
    f.close()
    ite = int(len(file_name)/batch_size)
    print('number of iteration :', ite)
    ## write file
    with open(name, 'w', newline='') as csvfile:
        c = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(["fname","label"])
        for cpt in range(0,ite):
        ## load data to test
            X = get_raw(file_name[cpt*batch_size:cpt*batch_size+batch_size], nsamples=n_samples, path=path_test)
            ## prediction
    
            proba = model.predict_on_batch(X)
            newproba = np.zeros((proba.shape[0], 12))
            newproba[:, 0:11] = proba[:, 0:11]
            newproba[:, 11] = proba[:, 11:].sum(axis=1)
            pred = np.argmax(newproba, axis=1)
            pred[pred>=11] = 11
            for i,p in enumerate(pred):
                c.writerow([file_name[cpt*batch_size+i], convert_pred(p)])           
def __main__():
    name_model = 'DeepModelVGGminus_CONV3_sgd_data_aug_relu.hdf5'#'DeepModelVGGConv15_Like-18.hdf5'#'DeepModelVGGConv9_minusBN-17.hdf5' #'DeepModelVGG_minusBN.hdf5'#'speech_model_mobilnet'
    file = 'test_kaggle.txt'
    path ='test/audio/'
    name = 'submission_DeepModelVGGminus_CONV3_sgd_data_aug_relu_nsamples16000.csv'
    write_result(name_model, file, path, name, n_samples=16000)
with tf.device('/gpu:0'):
    __main__()
