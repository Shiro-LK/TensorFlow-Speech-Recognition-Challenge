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
from collections import Counter
from scipy.io import wavfile
from preprocess_spectogram import data_augmentation
from model_DeepSpeech import get_model_audio, get_model_audio2, vgg_style,vgg_like, vgg14_audio, vgg14LSTM_audio, vgg14minusBN_audio
import keras
from keras import backend as K
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint
from keras.layers import SeparableConv2D, Conv1D, BatchNormalization, concatenate, GlobalMaxPool1D, MaxPooling1D, Dense, Input, Dropout, Flatten, Reshape, Activation, GlobalAveragePooling1D
from keras.layers.convolutional import SeparableConvolution2D
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.layers import LSTM, ConvLSTM2D
from keras.utils.generic_utils import CustomObjectScope

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

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
def convert_pred(pred):
    '''
        convert prediction number to prediction in string
    ''' 
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silent', 'unknown']
    return train_words[pred]

  
def prepare_data(filename):
    '''
        load file which give path and label for the data
    '''
    f = open(filename, 'r')
    
    data = [line.split() for line in f]
    feat =[]
    label=[]
    for l in data:
        feat.append(l[0])
        label.append(l[1])
   
    count = Counter(label)
    print(count)
    label = np.array(label, dtype=np.int)
    return feat, label

    
def batch_generator(batch_size, feat_path, labels, funct, nsample=16000, data_aug=False, proba=0.5, coeff_noise=0.25, coeff_time=4000):
    '''
        create a generator which load the sample given the path and the label. (for the fit function)
    '''
    print('generator:', 'data_aug=', data_aug,  'nsample=', nsample, 'proba_data_aug=', proba, 'coeff_noise=', coeff_noise, 'coeff_time=', coeff_time)
    number_classes = len(set(labels))
    labels = keras.utils.to_categorical(labels, num_classes=number_classes)
    n_sample = len(feat_path)
    ite = math.ceil(n_sample/batch_size)
    #print('iteration to do :', ite)
    while True:
        for i in range(0, ite):
            if i == ite-1:
                label = labels[-batch_size:]
                feat = funct(feat_path[-batch_size:], nsamples=nsample, data_aug=data_aug, proba=proba, coeff_noise=coeff_noise, coeff_time=coeff_time)
                #print('\nlast step\n')
                yield np.asarray(feat), label
            else:
                label = labels[i*batch_size:i*batch_size+batch_size]
                feat = funct(feat_path[i*batch_size:i*batch_size+batch_size], data_aug=data_aug, nsamples=nsample, proba=proba, coeff_noise=coeff_noise, coeff_time=coeff_time)
                yield np.asarray(feat), label

def batch_generator_shuffle(batch_size, feat_path, labels, funct, nsample=16000, data_aug=False, proba=0.5, coeff_noise=0.25, coeff_time=4000):
    '''
        create a generator which load the sample given the path and the label. (for the fit function)
    '''
    print('generator shuffle :', 'data_aug=', data_aug,  'nsample=', nsample, 'proba_data_aug=', proba, 'coeff_noise=', coeff_noise, 'coeff_time=', coeff_time)
    number_classes = len(set(labels))
    labels = keras.utils.to_categorical(labels, num_classes=number_classes)
    n_sample = len(feat_path)
    #print('iteration to do :', ite)
    while True:
        x = np.random.randint(labels.shape[0]-1, size=batch_size)
        paths = [feat_path[j] for j in x]
        
        label = labels[x,:]#labels[i*batch_size:i*batch_size+batch_size]
        feat = funct(paths, nsamples=nsample, data_aug=data_aug, proba=proba, coeff_noise=coeff_noise, coeff_time=coeff_time)#funct(feat_path[i*batch_size:i*batch_size+batch_size])
        yield np.asarray(feat), label


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
    
    
    train_generator = batch_generator_shuffle(batch_size, x_train, y_train, get_raw, data_aug=True, proba=0.6, coeff_noise=0.25, coeff_time=4000)
    
    test_generator = batch_generator(batch_size, x_test, y_test, get_raw, data_aug=False, proba=0.5, coeff_noise=0.25, coeff_time=4000)
    
    step_train = math.ceil(len(x_train)/batch_size)
    print('step train :', step_train)
    step_test = math.ceil(len(x_test)/batch_size)
   
    print('shape:', len(x_train))
    print('shape:', len(x_test))
    print('step train :' , step_train)
    print('step test :' , step_test)
    
    #  network 
    #model =vgg14LSTM_audio((16000,1), num_classes, 3, 500)#vgg_style((16000,1), num_classes, 15)
    #model = load_model('DeepModelVGG_CONV3_sgd.hdf5')

   # model = vgg_style((16000,1), num_classes, 3)    
    model = vgg14minusBN_audio((16000,1), num_classes, 3) #load_model('DeepModelVGGBigLSTM.hdf5')#
#    with CustomObjectScope({'f1' :f1}):
#        with tf.device('/cpu:0'):
#           oldmodel = load_model('DeepModelVGGConv9_minusBN-33.hdf5')
#    model = copy_weight(model, oldmodel)
    #model.save('DeepModelVGGBigBatchNorm.hdf5')
    #model = load_model('DeepModelVGGBigBatchNorm.hdf5')
    model.summary()
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adamax = keras.optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # 0.001 BIG
    adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-8, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
    
    with open(output+'.json','w') as f:
        json_string = model.to_json()
        f.write(json_string)
        
    # callback
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/'+output, histogram_freq=0, 
                                                       batch_size=batch_size, write_graph=True, write_grads=False, 
                                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                                                       embeddings_metadata=None)


    lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0000001)
    checkpoints = ModelCheckpoint(output+'.hdf5', verbose=1, save_best_only=True, period=checkpoint, save_weights_only=False)
    callbacks_list = [callback_tensorboard, checkpoints, lr_decay]
    
    # train 
    model.fit_generator(train_generator,
          steps_per_epoch=step_train,
          epochs=epochs,
          verbose=1,
          
          validation_data=test_generator,
          validation_steps=step_test,
          callbacks=callbacks_list)
    
def main():
    file_train = 'data_labelised2_train85.txt'#'data_labelised2_train09v2.txt'
    file_test = 'data_labelised2_test85.txt'#'data_labelised2_test09v2.txt'
    #data_train, label = prepare_data(file_train)
    #print(len(data_train))
    #gen = batch_generator(50, data_train, label, get_raw)
    #a = next(gen)
    #print(a[0].shape, a[1].shape)
    train_with_generator(path='', file=file_train, file_test= file_test, output='DeepModelVGGminus_CONV3_sgd_data_aug_relu', epochs=50, batch_size=32, checkpoint=1, shuffle=True)
with tf.device('/gpu:0'):
    main()
    
    
