# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:51:14 2017

@author: shiro
"""


import tensorflow as tf
import os
from scipy import signal
import math
import random
import numpy as np
from preprocess import load_data_with_spectrogram, load_data_with_mel_spectrogram
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
import pickle

def relu6(x):
    return K.relu(x, max_value=6) 
    
def convert_pred(pred):
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silence', 'unknown']
    return train_words[pred]

def get_raw(paths, nsamples=16000, resample=False, nresample=8000, path=''):
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
        if resample == True:
            d = signal.resample(d, nresample)
        data.append(d)


       
    return np.asarray(data, dtype=np.int16).reshape(T, -1, 1) #format conv 1D

    
def write_result_raw(name_model, file_test, path_test, name, batch_size=6, n_samples=16000, num_class=31):
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
    
    Y = np.zeros( (len(file_name), num_class) , dtype=np.float32)
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

def write_result_raw2(name_model, file_test, path_test, name, batch_size=6, n_samples=16000, num_class=31, resample=False):
    '''
        write submission for deep model.
        save also the prediction in pickle format.
    '''
    print('load json')
    model = load_model(name_model)#load_model(name_model+'-'+epoch+'.hdf5')
    f = open(file_test, 'r')
    
    file_name = [line.split()[0] for line in f]
    print(file_name[0:10])
    f.close()
    ite = int(math.ceil(len(file_name)/batch_size))
    print('number of iteration :', ite)
    
    Y = np.zeros( (len(file_name), num_class) , dtype=np.float32)
    ## write file
    with open(name, 'w', newline='') as csvfile:
        c = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(["fname","label"])
        for cpt in range(0,ite):
        ## load data to test
            if cpt%1000 == 0:
                print('[',cpt,'/',ite,']')
            if cpt*batch_size+batch_size < len(file_name):
                X = get_raw(file_name[cpt*batch_size:cpt*batch_size+batch_size], resample=resample, nsamples=n_samples, path=path_test)
                proba = model.predict_on_batch(X)
                Y[cpt*batch_size:cpt*batch_size+batch_size, :] = proba
            else:
                X = get_raw(file_name[cpt*batch_size:], resample=resample, nsamples=n_samples, path=path_test)
                proba = model.predict_on_batch(X)
                Y[cpt*batch_size:, :] = proba
            ## prediction
            #print(X.shape)
            
            pred = np.argmax(proba, axis=1)
            pred[pred>=11] = 11
            for i,p in enumerate(pred):
                c.writerow([file_name[cpt*batch_size+i], convert_pred(p)])   
    pickle.dump(Y, open(name_model.replace('.hdf5', '.pkl'), 'wb'))   

def write_result_spectrogram(name_model, file_test, path_test, name, batch_size=6, n_samples=16000, num_class=31, window_size=20, step_size=10, eps=1e-10):
    ## load model
    '''
        write submission for simple spectrogram.
        save also the prediction in pickle format.
    '''
    model = load_model(name_model)
    f = open(file_test, 'r')
    file_name = [line.split()[0] for line in f]
    f.close()
    model.summary()
    Y = np.zeros( (len(file_name), num_class) , dtype=np.float32)
    
    
    ite = int(math.ceil(len(file_name)/batch_size))
    print('number of iteration :', ite)
    ## write file
    with open(name, 'w', newline='') as csvfile:
        c = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(["fname","label"])
        for cpt in range(0,ite):
        ## load data to test
            if cpt%1000 == 0:
                print('[',cpt,'/',ite,']')
                
            if cpt*batch_size+batch_size < len(file_name):
                X = load_data_with_spectrogram(file_name[cpt*batch_size:cpt*batch_size+batch_size], nsamples=n_samples,p=path_test, window_size=window_size, step_size=step_size, eps=eps)
                proba = model.predict_on_batch(X)
                Y[cpt*batch_size:cpt*batch_size+batch_size, :] = proba
            else:
                X = load_data_with_spectrogram(file_name[cpt*batch_size:], nsamples=n_samples,p=path_test, window_size=window_size, step_size=step_size, eps=eps,)
                proba = model.predict_on_batch(X)
                Y[cpt*batch_size:, :] = proba
            #X = load_data_with_spectrogram(file_name[cpt*batch_size:cpt*batch_size+batch_size], p=path_test)
            ## prediction
    
            proba = model.predict_on_batch(X)
            pred = np.argmax(proba, axis=1)
            pred[pred>=11] = 11
            for i,p in enumerate(pred):
                c.writerow([file_name[cpt*batch_size+i], convert_pred(p)])   
        pickle.dump(Y, open(name_model.replace('.hdf5', '.pkl'), 'wb')) 

def write_result_spectrogram_mel(name_model, file_test, path_test, name, batch_size=6, n_samples=16000, n_mels=40, transpose=True, num_class=31):
    ## load model
    '''
        write submission for simple spectrogram.
        save also the prediction in pickle format.
    '''
    model = load_model(name_model)
    f = open(file_test, 'r')
    file_name = [line.split()[0] for line in f]
    f.close()
    
    Y = np.zeros( (len(file_name), num_class) , dtype=np.float32)
    
    
    ite = int(math.ceil(len(file_name)/batch_size))
    print('number of iteration :', ite)
    ## write file
    with open(name, 'w', newline='') as csvfile:
        c = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(["fname","label"])
        for cpt in range(0,ite):
        ## load data to test
            if cpt%1000 == 0:
                print('[',cpt,'/',ite,']')
                
            if cpt*batch_size+batch_size < len(file_name):
                X = load_data_with_mel_spectrogram(file_name[cpt*batch_size:cpt*batch_size+batch_size], transpose=transpose, n_mels=n_mels, nsamples=n_samples,p=path_test)
                proba = model.predict_on_batch(X)
                Y[cpt*batch_size:cpt*batch_size+batch_size, :] = proba
            else:
                X = load_data_with_mel_spectrogram(file_name[cpt*batch_size:], transpose=transpose, n_mels=n_mels,nsamples=n_samples,p=path_test)
                proba = model.predict_on_batch(X)
                Y[cpt*batch_size:, :] = proba
            #X = load_data_with_spectrogram(file_name[cpt*batch_size:cpt*batch_size+batch_size], p=path_test)
            ## prediction
    
            proba = model.predict_on_batch(X)
            pred = np.argmax(proba, axis=1)
            pred[pred>=11] = 11
            for i,p in enumerate(pred):
                c.writerow([file_name[cpt*batch_size+i], convert_pred(p)])   
        pickle.dump(Y, open(name_model.replace('.hdf5', '.pkl'), 'wb')) 
                
def write_resultIMG(name_model, epoch, file_test, path_test, name, batch_size=6,n_samples=16000):
    '''
        write submission keras for mobilenet model
    '''
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
                
def write_result_bagging(name_model, file_test, path_test, name):
    ## load model
    '''
        write submission for simple spectrogram from pickle file.
        save also the prediction in pickle format.
    '''
    Y = np.zeros((158538,12), dtype=np.float32)
    for model in name_model:
        pred = pickle.load(open(model, 'rb'))
        Y += pred
    pred = np.argmax(Y, axis=1)
    pred[pred>=11] = 11
    
    f = open(file_test, 'r')
    file_name = [line.split()[0] for line in f]
    print(file_name[0:10])
    f.close()
    
    ite = 158538
    print('number of iteration :', ite)
    ## write file
    with open(name, 'w', newline='') as csvfile:
        c = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(["fname","label"])
        for cpt in range(0,ite):
        ## load data to test
            if cpt%1000 == 0:
                print('[',cpt,'/',ite,']')
                
            
            #X = load_data_with_spectrogram(file_name[cpt*batch_size:cpt*batch_size+batch_size], p=path_test)
            ## prediction
            c.writerow([file_name[cpt], convert_pred(pred[cpt])])   
        pickle.dump(Y, open(name.replace('.csv', '.pkl'), 'wb')) 
def __main__():
    #name_model = 'VGG14_raw_eq_data_aug.hdf5' #'DeepModelVGGminus_CONV3_sgd_data_aug_relu.hdf5'#'DeepModelVGGConv15_Like-18.hdf5'#'DeepModelVGGConv9_minusBN-17.hdf5' #'DeepModelVGG_minusBN.hdf5'#'speech_model_mobilnet'
    name_model = 'SpeechGetModel2_conv3_spectrogram_data_aug_relu_2FC.hdf5'    
    file = 'test_kaggle.txt'
    path ='test/audio/'
    bagging = ['modele_12/mfcc/SpeechModelaudio2_Conv3_spectrogramMFCC_eq_data_aug_ampl.pkl', 'modele_12/mfcc/SpeechSmallVGG_Conv3_spectrogramMFCC_eq_data_aug_ampl.pkl']
    name2 = 'submission_VGGstyle_raw_eq_data_aug.csv'
    
    name_model2 = 'VGGstyle_raw_eq_data_aug.hdf5' 
    name = 'submission_SpeechGetModel2_conv3_spectrogram_data_aug_relu_2FC.csv'
    write_result_spectrogram(name_model, file, path, name, batch_size=128, n_samples=16000, num_class=12, window_size=20)
    #write_result_spectrogram_mel(name_model, file, path, name,batch_size=100, n_samples=16000, num_class=12)
    
    #write_result_raw2(name_model2, file, path, name2, batch_size=64, n_samples=16000, num_class=12, resample=True)
    #write_result_bagging(bagging, file, path, 'bagging.csv')
with tf.device('/gpu:0'):
    __main__()
