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
from sklearn.metrics import accuracy_score
from collections import Counter
from scipy.io import wavfile
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
from preprocess_spectogram import load_data_with_spectrogram
def relu6(x):
    return K.relu(x, max_value=6) 
    
def convert_pred(pred):
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silence', 'unknown']
    return train_words[pred]
def get_specgrams2D(paths, nsamples=16000, path=''):
    '''
    Given list of paths, return specgrams.
    '''
    
    # read the wav files
    wavs = []
    for x in paths:
        temp = wavfile.read(path+x)[1]
    # zero pad the shorter samples and cut off the long ones.
    data = [] 
    
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
        else:
            d = wav[0:nsamples]
        data.append(d)

    # get the specgram
    specgram = [signal.spectrogram(d, nperseg=256, noverlap=128)[2] for d in data]
    specgram = [np.repeat(s.reshape(129, 124)[:, :, np.newaxis], 3, axis=2) for s in specgram]
    
    return np.asarray(specgram)
    
def write_result(name_model, epoch, file_test, path_test, name, batch_size=6):
    ## load model
    #model = model_from_json(name_model+'.json')
    print('load json')
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(name_model+'.hdf5')
    f = open(file_test, 'r')
    
    file_name = [line.split()[0] for line in f]
    #print(file_name[0:10])
    f.close()
    ite = int(len(file_name)/batch_size)
    print('number of iteration :', ite)
    ## write file
    with open(name, 'w', newline='') as csvfile:
        c = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(["fname","label"])
        for cpt in range(0,ite):
        ## load data to test
            X = get_specgrams2D(file_name[cpt*batch_size:cpt*batch_size+batch_size], path=path_test)
            ## prediction
    
            proba = model.predict_on_batch(X)
            pred = np.argmax(proba, axis=1)
            pred[pred>=11] = 11
            for i,p in enumerate(pred):
                c.writerow([file_name[cpt*batch_size+i], convert_pred(p)])   
def write_result_spectrogram(name_model, epoch, file_test, path_test, name, batch_size=6):
    ## load model
    #model = model_from_json(name_model+'.json')
    print('load json')
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(name_model+'.hdf5')
    f = open(file_test, 'r')
    
    file_name = [line.split()[0] for line in f]
    #print(file_name[0:10])
    f.close()
    ite = int(len(file_name)/batch_size)
    print('number of iteration :', ite)
    ## write file
    with open(name, 'w', newline='') as csvfile:
        c = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(["fname","label"])
        for cpt in range(0,ite):
        ## load data to test
            if cpt%1000 == 0:
                print('[',cpt,'/',ite,']')
            X = load_data_with_spectrogram(file_name[cpt*batch_size:cpt*batch_size+batch_size], p=path_test)
            ## prediction
    
            proba = model.predict_on_batch(X)
            pred = np.argmax(proba, axis=1)
            pred[pred>=11] = 11
            for i,p in enumerate(pred):
                c.writerow([file_name[cpt*batch_size+i], convert_pred(p)])           
def __main__():
    name_model = 'DeepModel_Conv9_spectrogram_data_aug'#'speech_model_mobilnet'
    epoch = '03'
    file = 'test_kaggle.txt'
    path ='test/audio/'
    name = 'submissionTest_DeepModel_Conv9_spectrogram_data_aug.csv'
    
    name_model2 = 'DeepModelVGGminus_CONV3_sgd_data_aug'
    name2 = 'submissionTest_'+name_model2+'.csv'
    write_result(name_model2, epoch, file, path, name2)
    #write_result_spectrogram(name_model, epoch, file, path, name)
with tf.device('/gpu:0'):
    __main__()