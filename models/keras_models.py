# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:41:39 2022

@author: unknown
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout


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

