# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:57:54 2017

@author: shiro
"""
import tensorflow as tf
import os
from scipy import signal
import math
import random
import numpy as np
from preprocess import load_data_with_spectrogram, prepare_data, load_data_with_mel_spectrogram
from collections import Counter
from scipy.io import wavfile
import keras
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint
from keras.layers import SeparableConv2D, Conv1D, BatchNormalization, concatenate, GlobalMaxPool1D, MaxPooling1D, Dense, Input, Dropout, Flatten, Reshape, Activation, GlobalAveragePooling1D
from keras.layers.convolutional import SeparableConvolution2D
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import LSTM, ConvLSTM2D, GlobalMaxPooling1D
from keras.utils import np_utils
from model_DeepSpeech import get_model_audio, get_model_audio2, vgg_style,vgg_like, vgg14_audio, vgg14LSTM_audio, vgg14minusBN_audio
def convert_pred(pred):
    '''
        convert prediction number to prediction in string
    ''' 
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silent', 'unknown']
    return train_words[pred]


       
def batch_generator(batch_size, feat_path, labels, funct):
    '''
        create a generator which load the sample given the path and the label. (for the fit function)
    '''
    number_classes = len(set(labels))
    labels = keras.utils.to_categorical(labels, num_classes=number_classes)
    n_sample = len(feat_path)
    ite = math.ceil(n_sample/batch_size)
    #print('iteration to do :', ite)
    while True:
        for i in range(0, ite):
            if i == ite-1:
                label = labels[-batch_size:]
                feat = funct(feat_path[-batch_size:])
                #print('\nlast step\n')
                yield np.asarray(feat), label
            else:
                label = labels[i*batch_size:i*batch_size+batch_size]
                feat = funct(feat_path[i*batch_size:i*batch_size+batch_size])
                yield np.asarray(feat), label

def batch_generator_shuffle(batch_size, feat_path, labels, funct, window_size=20, step_size=10, eps=1e-10, data_aug=False, proba_data_aug=0.5, coeff_noise=0.3, coeff_time=4000):
    '''
        create a generator which load the sample given the path and the label. (for the fit function)
    '''
    print('window_size=', window_size, 'step_size=' , step_size, 'eps=', eps, 'data_aug=', data_aug,  'proba_data_aug=', proba_data_aug, 'coeff_noise=', coeff_noise, 'coeff_time=', coeff_time)
    num_classes = len(set(labels))
    #print('iteration to do :', ite)
    while True:
        index= np.random.randint(len(feat_path)-1, size=batch_size)
        #print(index)
        feat = [feat_path[i] for i in index]
        batch_features = funct(feat, window_size=window_size, step_size=step_size, eps=eps, data_aug=data_aug, proba_data_aug=proba_data_aug, coeff_noise=coeff_noise, coeff_time=coeff_time)
        batch_labels = np_utils.to_categorical(labels[index], num_classes)
        yield batch_features, batch_labels
        

def copy_weight(newmodel, oldmodel):
    dic_w = {}
    for layer in oldmodel.layers:
        dic_w[layer.name] = layer.get_weights()
    
    for layer in newmodel.layers:
        if layer.name in dic_w:
            layer.set_weights(dic_w[layer.name])
            print(layer.name)
    return newmodel

def train_with_generator(path, file, file_test, output, epochs, batch_size, checkpoint, shuffle=False):
    
    # prepare data
    x_train, y_train  =  prepare_data(file)
    x_test, y_test  =  prepare_data(file_test)
    num_classes = len(list(set(y_train)))
    print('nombre de classes : ', num_classes)
    #x_train, x_test, y_train , y_test = train_test_split(X, Y, test_size=0.2)#, stratify=y)
    
    
    #train_generator = batch_generator_shuffle(batch_size, x_train, y_train, load_data_with_spectrogram, window_size=20, step_size=10, eps=1e-10, data_aug=True, proba_data_aug=0.6, coeff_noise=0.3, coeff_time=4000)

    train_generator = batch_generator_shuffle(batch_size, x_train, y_train, load_data_with_mel_spectrogram, window_size=20,
                                              step_size=10, eps=1e-10, data_aug=True, proba_data_aug=0.6,
                                              coeff_noise=0.3)

    test_generator = batch_generator(batch_size, x_test, y_test, load_data_with_spectrogram)
    
    step_train = math.ceil(len(x_train)/batch_size)
    print('step train :', step_train)
    step_test = math.ceil(len(x_test)/batch_size)
   
    print('shape:', len(x_train))
    print('shape:', len(x_test))
    print('step train :' , step_train)
    print('step test :' , step_test)
    
    #  network 
    model = get_model_audio((99,161), num_classes, 9)#vgg_style((99,161), num_classes)
    model.summary()
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
#    
#    with open(output+'.json','w') as f:
#        json_string = model.to_json()
#        f.write(json_string)
        
    # callback
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/'+output, histogram_freq=0, 
                                                       batch_size=batch_size, write_graph=True, write_grads=False, 
                                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                                                       embeddings_metadata=None)
    # -{epoch:02d}
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    checkpoints = ModelCheckpoint(output+'.hdf5', verbose=1, save_best_only=True, period=checkpoint, save_weights_only=False)
    callbacks_list = [callback_tensorboard, checkpoints, reduce]
    
    # train 
    model.fit_generator(train_generator,
          steps_per_epoch=step_train,
          epochs=epochs,
          verbose=1,
          
          validation_data=test_generator,
          validation_steps=step_test,
          callbacks=callbacks_list)
    
def main():
    file_train = 'data_labelised2_train85.txt'#'data_labelised3_equilibrate_train80.txt'#'data_labelised2_train09.txt'
    file_test = 'data_labelised2_test85.txt'#'data_labelised3_equilibrate_test80.txt'#'data_labelised2_test09.txt'
    
    train_with_generator(path='', file=file_train, file_test= file_test, output='DeepModel_Conv9_spectrogram_data_aug', epochs=40, batch_size=128, checkpoint=1, shuffle=True)
with tf.device('/gpu:0'):
    main()
    
    