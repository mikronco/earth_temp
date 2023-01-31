# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:41:39 2022

@author: unknown
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU, Dropout, Concatenate




class FCNNRain():
    def __init__(self):
        super().__init__()
        
    def initialize(self, nin, nhl1, nhl2, activ):
        model = Sequential()
        model.add(Dense(nhl1, input_dim= nin, activation=activ))
        model.add(Dense(nhl2, activation=activ))
        model.add(Dense(1, activation='sigmoid'))
        return model 


class FCNNRain_Dropout():
    def __init__(self):
        super().__init__()
        
    def initialize(self, nin, nhl1, nhl2, nhl3, nhl4, activ, dp):
        model = Sequential()
        model.add(Dense(units=nhl1,input_dim=nin,activation=activ))
        model.add(Dense(units=nhl2,activation=activ))
        model.add(Dropout(dp))
        model.add(Dense(units=nhl3,activation=activ))
        model.add(Dropout(dp))
        model.add(Dense(units=nhl4,activation=activ))
        model.add(Dense(1, activation='sigmoid'))
        return model 


class FCNNRain_Dropout_flex():
    def __init__(self):
        super().__init__()
        
    def initialize(self, activ, dp, nin, *nhl):
        model = Sequential()
        model.add(Dense(units=nhl[0],input_dim=nin,activation=activ))
        for n in nhl[1:]:
            model.add(Dense(units=n,activation=activ))
            model.add(Dropout(dp))
        model.add(Dense(1, activation='sigmoid'))
        return model 



class CNNStorm():
    def __init__(self):
        super().__init__()
        
    def initialize(self, input_shape, num_conv_filters, filter_width, conv_activation):
        conv_net_in = Input(shape=input_shape[1:])
        conv_net = Conv2D(num_conv_filters, (filter_width, filter_width), padding="same")(conv_net_in)
        conv_net = Activation(conv_activation)(conv_net)
        conv_net = AveragePooling2D()(conv_net)
        conv_net = Conv2D(num_conv_filters * 2, (filter_width, filter_width), padding="same")(conv_net)
        conv_net = Activation(conv_activation)(conv_net)
        conv_net = AveragePooling2D()(conv_net)
        conv_net = Flatten()(conv_net)
        conv_net = Dense(1)(conv_net)
        conv_net = Activation("sigmoid")(conv_net)
        conv_model = Model(conv_net_in, conv_net)
        
        return conv_model


class CNNIce():
    def __init__(self):
        super().__init__()
        
    def initialize(self, input_shape, num_conv_filters, filter_width, conv_activation):
        conv_net_in = Input(shape=input_shape[1:])
        conv_net = Conv2D(num_conv_filters, (filter_width, filter_width), padding="same")(conv_net_in)
        conv_net = Activation(conv_activation)(conv_net)
        conv_net = AveragePooling2D()(conv_net)
        conv_net = Conv2D(num_conv_filters * 2, (filter_width, filter_width), padding="same")(conv_net)
        conv_net = Activation(conv_activation)(conv_net)
        conv_net = AveragePooling2D()(conv_net)
        conv_net = Flatten()(conv_net)
        conv_net = Dense(1)(conv_net)
        conv_model = Model(conv_net_in, conv_net)
        
        return conv_model


class CNNIce1():
    def __init__(self):
        super().__init__()
        
    def initialize(self, img_shape, vec_shape, num_conv_filters, filter_width, conv_activation):
        conv_net_in = Input(shape=img_shape[1:])
        vector_input = Input(shape=vec_shape[1:])
        fc_net = Activation(conv_activation)(vector_input)

        conv_net = Conv2D(num_conv_filters, (filter_width, filter_width), padding="same")(conv_net_in)
        conv_net = Activation(conv_activation)(conv_net)
        conv_net = AveragePooling2D()(conv_net)
        conv_net = Conv2D(num_conv_filters * 2, (filter_width, filter_width), padding="same")(conv_net)
        conv_net = Activation(conv_activation)(conv_net)
        conv_net = AveragePooling2D()(conv_net)
        conv_net = Flatten()(conv_net)
        concat_layer= Concatenate()([conv_net, fc_net])
        conv_net = Dense(1)(conv_net)

        conv_model = Model([conv_net_in, vector_input], conv_net)
        
        return conv_model


