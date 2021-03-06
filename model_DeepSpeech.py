# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:32:06 2018

@author: Shiro
"""
import keras
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint
from keras.layers import SeparableConv2D, Conv1D, BatchNormalization, concatenate, GlobalMaxPool1D, MaxPooling1D, Dense, Input, Dropout, Flatten, Reshape, Activation, GlobalAveragePooling1D
from keras.layers.convolutional import SeparableConvolution2D
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import LSTM, ConvLSTM2D, GlobalMaxPooling1D, MaxPooling2D, GlobalMaxPooling2D



### ---- Model which works ---- ####


## spectrogram 0.815
def get_model1(shape, num_class, kernel_size=3, act='elu'):
    '''Create a keras model. average VGG14'''
    inputlayer = Input(shape=shape)
    
    model = Conv1D(64, kernel_size , activation=act, name='conv4', padding='same')(inputlayer)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(128, kernel_size , activation=act, name='conv5', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    model = GlobalMaxPooling1D()(model)
    #model = Flatten()(model)
    model = Dense(1024, activation='relu', name='dense1')(model)
    model = Dropout(0.5)(model)
    model = Dense(1024, activation='relu', name='dense2')(model)
    model = Dropout(0.5)(model)
#    
    # 11 because background noise has been taken out
    model = Dense(num_class, activation='softmax', name='soft')(model)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model
    
    
## MFCC 0.85
def get_model2(shape, num_class, kernel_size=3, act='relu'):
    '''Create a keras model. average VGG14'''
    inputlayer = Input(shape=shape)
    
    model = Conv1D(256, kernel_size , activation=act, name='conv4', padding='same')(inputlayer)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(512, kernel_size , activation=act, name='conv5', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    model = GlobalMaxPooling1D()(model)
    #model = Flatten()(model)
    model = Dense(1024, activation='relu', name='dense1')(model)
    model = Dropout(0.5)(model)
    model = Dense(1024, activation='relu', name='dense2')(model)
    model = Dropout(0.5)(model)
#    
    # 11 because background noise has been taken out
    model = Dense(num_class, activation='softmax', name='soft')(model)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model
    
    
    
## spectrogram 0,81   
def small_vgg(shape, num_class, kernel_size=3, act='relu'):
    inputlayer = Input(shape=shape)
    
    x = Conv1D(64, kernel_size , activation=act, name='conv64_1', padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size , activation=act, name='conv64_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    x = Conv1D(128, kernel_size , activation=act, name='conv128_1', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size , activation=act, name='conv128_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    
    
    x = GlobalMaxPooling1D()(x)
    x = Dense(1024, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', name='dense2')(x)
    x = Dropout(0.5)(x)
#    
    # 11 because background noise has been taken out
    x = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=x)
    
    return model

  ## MFCC 0.85  
def small_vgg(shape, num_class, kernel_size=3, act='relu'):
    inputlayer = Input(shape=shape)
    
    x = Conv1D(64, kernel_size , activation=act, name='conv64_1', padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size , activation=act, name='conv64_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    x = Conv1D(128, kernel_size , activation=act, name='conv128_1', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size , activation=act, name='conv128_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    
    
    x = GlobalMaxPooling1D()(x)
    x = Dense(2048, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='dense2')(x)
    x = Dropout(0.5)(x)
#    
    # 11 because background noise has been taken out
    x = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=x)
    
    return model 
## raw 0.75 after 30 epochs
def vgg14_audio(shape, num_class, kernel_size=3):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)
    x = inputlayer 
    # Block 1
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block1_conv1')(x)
    x = BatchNormalization()(x)    
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)      
    x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block2_conv2')(x) 
    x = MaxPooling1D(2, strides=2, name='block2_pool')(x)
    x = BatchNormalization()(x) 

    # Block 3
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block3_pool')(x)
    x = BatchNormalization()(x) 

    # Block 4
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block4_pool')(x)
    x = BatchNormalization()(x) 

    # Block 5
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv3')(x)  
    x = MaxPooling1D(2, strides=2, name='block5_pool')(x)
    #x = Flatten(name='flatten')(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x) 
    # classif
    x = Dense(1024, activation = 'relu', name= 'dense_1024')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu', name= 'dense2_1024')(x)
    x = Dropout(0.5)(x)
    
    model = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model
    

## TO TEST
def vggnet1D(shape, num_class, kernel_size=3, act='relu'):
    inputlayer = Input(shape=shape)
    
    x = Conv1D(8, kernel_size , activation=act, name='conv8_1', padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = Conv1D(8, kernel_size , activation=act, name='conv8_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    x = Conv1D(16, kernel_size , activation=act, name='conv16_1', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(16, kernel_size , activation=act, name='conv16_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    x = Conv1D(32, kernel_size , activation=act, name='conv32_1', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size , activation=act, name='conv32_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    x = GlobalMaxPooling1D()(x)
    x = Dense(1024, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='dense2')(x)
    x = Dropout(0.5)(x)
#    
    # 11 because background noise has been taken out
    x = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=x)
    
    return model
def vggnet2D(shape, num_class, kernel_size=3, act='relu'):
    inputlayer = Input(shape=shape)
    
    x = SeparableConv2D(8, kernel_size , activation=act, name='conv8_1', padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = SeparableConv2D(8, kernel_size , activation=act, name='conv8_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2))(x)
    
    x = SeparableConv2D(16, kernel_size , activation=act, name='conv16_1', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(16, kernel_size , activation=act, name='conv16_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2))(x)
    
    x = SeparableConv2D(32, kernel_size , activation=act, name='conv32_1', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(32, kernel_size , activation=act, name='conv32_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2))(x)
    
    x = GlobalMaxPooling2D()(x)
    x = Dense(1024, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='dense2')(x)
    x = Dropout(0.5)(x)
#    
    # 11 because background noise has been taken out
    x = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=x)
    
    return model
    
def vggnet1D_2(shape, num_class, kernel_size=3, act='relu'):
    inputlayer = Input(shape=shape)
    x = inputlayer
    x = Reshape((shape[0], shape[1], 1))(x)
    

    
    x = Conv1D(64, kernel_size , activation=act, name='conv64_1', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size , activation=act, name='conv64_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    x = Conv1D(128, kernel_size , activation=act, name='conv128_1', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size , activation=act, name='conv128_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    
    x = Conv1D(256, kernel_size , activation=act, name='conv256_1', padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size , activation=act, name='conv256_2', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D((2))(x)
    #x = Flatten()(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1024, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='dense2')(x)
    x = Dropout(0.5)(x)
#    
    # 11 because background noise has been taken out
    x = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=x)
    
    return model
    
def get_model_audio(shape, num_class, kernel_size=3, act='relu'):
    '''Create a keras model. mini VGG 14'''
    inputlayer = Input(shape=shape)

    #model = BatchNormalization()(inputlayer)
    model = Conv1D(8, kernel_size , activation=act, name='conv1', padding='same')(inputlayer)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(16, kernel_size , activation=act, name='conv2', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(32, kernel_size , activation=act, name='conv3', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(64, kernel_size , activation=act, name='conv4', padding='same')(inputlayer)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(128, kernel_size , activation=act, name='conv5', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)

    #model = BatchNormalization()(model)
    #model = Dropout(0.25)(model)
    #model = MaxPooling1D((2))(model)
    
    model = GlobalMaxPooling1D()(model)
    #model = Flatten()(model)
    model = Dense(1024, activation='relu', name='dense1')(model)
    model = Dropout(0.5)(model)
    model = Dense(1024, activation='relu', name='dense2')(model)
    model = Dropout(0.5)(model)
#    
    # 11 because background noise has been taken out
    model = Dense(num_class, activation='softmax', name='soft')(model)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model
    
def get_model_audio2(shape, num_class, kernel_size=3, act='relu'):
    '''Create a keras model. average VGG14'''
    inputlayer = Input(shape=shape)

    #model = BatchNormalization()(inputlayer)
    model = Conv1D(32, kernel_size , activation=act, name='conv1', padding='same')(inputlayer)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(64, kernel_size , activation=act, name='conv2', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(128, kernel_size , activation=act, name='conv3', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(256, kernel_size , activation=act, name='conv4', padding='same')(inputlayer)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)
    
    model = Conv1D(512, kernel_size , activation=act, name='conv5', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D((2))(model)

    #model = BatchNormalization()(model)
    #model = Dropout(0.25)(model)
    #model = MaxPooling1D((2))(model)
    
    model = GlobalMaxPooling1D()(model)
    #model = Flatten()(model)
    model = Dense(1024, activation='relu', name='dense1')(model)
    model = Dropout(0.5)(model)
    model = Dense(1024, activation='relu', name='dense2')(model)
    model = Dropout(0.5)(model)
#    
    # 11 because background noise has been taken out
    model = Dense(num_class, activation='softmax', name='soft')(model)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model



    
def vgg_style(shape, num_class, kernel_size=3, act='elu'):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)
    x = inputlayer
    for i in range(9): 
        # 8 16 32 64 128
        x = Conv1D(8*(2 ** i), (kernel_size),padding = 'same', name='conv'+str(i),)(x) 
        x = BatchNormalization()(x) 
        x = Activation(act)(x) 
        x = MaxPooling1D((2), padding='same')(x)
   
    x_1d_branch_1 = GlobalAveragePooling1D()(x)
    x_1d_branch_2 = GlobalMaxPool1D()(x)
    x = concatenate([x_1d_branch_1, x_1d_branch_2])
    
    x = Dense(1024, activation = 'relu', name= 'dense1024')(x)
    x = Dropout(0.4)(x)
    model = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model

def vgg_like(shape, num_class, kernel_size=3, kernel_lstm=1000):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)
    x = inputlayer
    for i in range(6): 
        # 8 16 32 64 128
        x = Conv1D(8*(2 ** i), (kernel_size), activation='relu', padding = 'same', name='conv'+str(i))(x) 
        x = BatchNormalization()(x)         
        #x = Activation('relu')(x) 
        x = MaxPooling1D((2), padding='same')(x)
        
    x = LSTM(kernel_lstm)
    #x_1d_branch_1 = GlobalAveragePooling1D()(x)
    #x_1d_branch_2 = GlobalMaxPool1D()(x)
    #x = concatenate([x_1d_branch_1, x_1d_branch_2])
    
    x = Dense(1024, activation = 'relu', name= 'dense1024')(x)
    x = Dropout(0.5)(x)
    model = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model

def vgg14minusBN_audio(shape, num_class, kernel_size=3):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)
    #x = BatchNormalization()(inputlayer)       
    # Block 1
    x = Conv1D(8, kernel_size, activation='relu', padding='same', name='block1_conv1')(inputlayer)
    x = BatchNormalization()(x)      
    x = Conv1D(8, kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)  
    x = MaxPooling1D(2, strides=2, name='block1_pool')(x)
    

    # Block 2
    x = Conv1D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(16, kernel_size, activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)     
    x = MaxPooling1D(2, strides=2, name='block2_pool')(x)
    
    
    # Block 3
    x = Conv1D(32, kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x) 
    x = Conv1D(32, kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x) 
    x = Conv1D(32, kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)     
    x = MaxPooling1D(2, strides=2, name='block3_pool')(x)
    
    
    # Block 4
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x) 
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x) 
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)     
    x = MaxPooling1D(2, strides=2, name='block4_pool')(x)
    

    # Block 5
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x) 
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x) 
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)     
    x = MaxPooling1D(2, strides=2, name='block5_pool')(x)
    #x = Flatten(name='flatten')(x)
    x = GlobalMaxPool1D()(x)
    
    #x = GlobalMaxPool1D()(x)  
    
    #x = LSTM(kernel_lstm, name='LSTM1')(x)
    # classif
    x = Dense(1024, activation = 'relu', name= 'dense_1')(x)
    x = Dropout(0.5)(x)
    #x = Dense(1024, activation = 'relu', name= 'dense_2')(x)
    #x = Dropout(0.5)(x)    
    model = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model

    

def vgg14LSTM_audio(shape, num_class, kernel_size=3, kernel_lstm=1000):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)
    x = BatchNormalization()(inputlayer) 
    # Block 1
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block1_conv1')(x)
    x = BatchNormalization()(x)    
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)      
    x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block2_conv2')(x) 
    x = MaxPooling1D(2, strides=2, name='block2_pool')(x)
    x = BatchNormalization()(x) 

    # Block 3
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block3_pool')(x)
    x = BatchNormalization()(x) 

    # Block 4
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block4_pool')(x)
    x = BatchNormalization()(x) 

    # Block 5
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv3')(x)  
    x = MaxPooling1D(2, strides=2, name='block5_pool')(x)
    #x = Flatten(name='flatten')(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x) 
    x = LSTM(kernel_lstm, name='LSTM1')(x)#GlobalMaxPool1D()(x) ConvLSTM2D
    # classif
    x = Dense(1024, activation = 'relu', name= 'dense_1024')(x)
    x = Dropout(0.5)(x)
    model = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model

def Deepvgg14_audio(shape, num_class, kernel_size=3):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)
    x = inputlayer 
    
    # Block 1
    x = Conv1D(8, kernel_size, activation='relu', padding='same', name='block0_conv1')(x)
    x = BatchNormalization()(x)    
    x = Conv1D(8, kernel_size, activation='relu', padding='same', name='block0_conv2')(x)
    x = BatchNormalization()(x)      
    x = MaxPooling1D(2, strides=2, name='block0_pool')(x)
    
    x = Conv1D(16, kernel_size, activation='relu', padding='same', name='block01_conv1')(x)
    x = BatchNormalization()(x)    
    x = Conv1D(16, kernel_size, activation='relu', padding='same', name='block01_conv2')(x)
    x = BatchNormalization()(x)      
    x = MaxPooling1D(2, strides=2, name='block01_pool')(x)
    
    x = Conv1D(32, kernel_size, activation='relu', padding='same', name='block02_conv1')(x)
    x = BatchNormalization()(x)    
    x = Conv1D(32, kernel_size, activation='relu', padding='same', name='block02_conv2')(x)
    x = BatchNormalization()(x)      
    x = MaxPooling1D(2, strides=2, name='block02_pool')(x)
    
    # Block 1
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block1_conv1')(x)
    x = BatchNormalization()(x)    
    x = Conv1D(64, kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)      
    x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(128, kernel_size, activation='relu', padding='same', name='block2_conv2')(x) 
    x = MaxPooling1D(2, strides=2, name='block2_pool')(x)
    x = BatchNormalization()(x) 

    # Block 3
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(256, kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block3_pool')(x)
    x = BatchNormalization()(x) 

    # Block 4
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)      
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block4_pool')(x)
    x = BatchNormalization()(x) 

    # Block 5
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(512, kernel_size, activation='relu', padding='same', name='block5_conv3')(x)  
    x = MaxPooling1D(2, strides=2, name='block5_pool')(x)
    
    # Block 6
    x = Conv1D(1024, kernel_size, activation='relu', padding='same', name='block6_conv1')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(1024, kernel_size, activation='relu', padding='same', name='block6_conv2')(x)
    x = BatchNormalization()(x)  
    x = Conv1D(1024, kernel_size, activation='relu', padding='same', name='block6_conv3')(x)  
    x = MaxPooling1D(2, strides=2, name='block6_pool')(x)
    
    
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x) 
    # classif
    x = Dense(1024, activation = 'relu', name= 'dense_1024')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu', name= 'dense2_1024')(x)
    x = Dropout(0.5)(x)
    model = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model
    
def vgg142D_audio(shape, num_class, kernel_size=3, act="relu"):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)
    x = inputlayer
    x = Reshape((shape[0], shape[1], 1))(x)
    # Block 1
    x = SeparableConv2D(64, kernel_size, activation=act, padding='same', name='block1_conv1')(x)
    x = BatchNormalization()(x)    
    x = SeparableConv2D(64, kernel_size, activation=act, padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)      
    x = MaxPooling2D(2, strides=2, name='block1_pool')(x)

    # Block 2
    x = SeparableConv2D(128, kernel_size, activation=act, padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)      
    x = SeparableConv2D(128, kernel_size, activation=act, padding='same', name='block2_conv2')(x) 
    x = MaxPooling2D(2, strides=2, name='block2_pool')(x)
    x = BatchNormalization()(x) 

    # Block 3
    x = SeparableConv2D(256, kernel_size, activation=act, padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)  
    x = SeparableConv2D(256, kernel_size, activation=act, padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)  
    x = SeparableConv2D(256, kernel_size, activation=act, padding='same', name='block3_conv3')(x)
    x = MaxPooling2D(2, strides=2, name='block3_pool')(x)
    x = BatchNormalization()(x) 

    # Block 4
    x = SeparableConv2D(512, kernel_size, activation=act, padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)      
    x = SeparableConv2D(512, kernel_size, activation=act, padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)      
    x = SeparableConv2D(512, kernel_size, activation=act, padding='same', name='block4_conv3')(x)
    x = MaxPooling2D(2, strides=2, name='block4_pool')(x)
    x = BatchNormalization()(x) 

    # Block 5
    x = SeparableConv2D(512, kernel_size, activation=act, padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)  
    x = SeparableConv2D(512, kernel_size, activation=act, padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)  
    x = SeparableConv2D(512, kernel_size, activation=act, padding='same', name='block5_conv3')(x)  
    #x = MaxPooling1D(2, strides=2, name='block5_pool')(x)
    #x = Flatten(name='flatten')(x)
    x = GlobalMaxPooling2D()(x)
    x = BatchNormalization()(x) 
    # classif
    x = Dense(1024, activation = 'relu', name= 'dense_1024')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu', name= 'dense2_1024')(x)
    x = Dropout(0.5)(x)
    
    model = Dense(num_class, activation='softmax', name='soft')(x)
    
    model = Model(inputs=inputlayer, outputs=model)
    
    return model    