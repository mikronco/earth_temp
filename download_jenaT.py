# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:02:10 2022

@author: unknown
"""

from zipfile import ZipFile
import os
import keras

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall(path="./data/")

