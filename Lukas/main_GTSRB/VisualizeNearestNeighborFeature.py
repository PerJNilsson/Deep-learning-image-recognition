from __future__ import print_function
from PIL import Image
import numpy as np
import csv
import os
import keras
from matplotlib import pyplot as plt
from keras.models import load_model
from readData import readTestData
from NNfunctions import NNPixel, NNFeature

img_x=img_y = 36
validation_size = 10
num_classes = 43
input_shape = (img_x, img_y, 3)

model_path = 'testcaseLukas_model'
model = load_model(model_path)

x_test, y_test = readTestData('C:\\Users\\nystr\\GTSRB\\Final_Test\\Images',
                                               (img_x, img_y), validation_size, num_classes,img_x, img_y)
#Now x_text and y_text are ready for the Model
choosen_index = 0
num_NN = 5
NNPixel(x_test,choosen_index,num_NN)  #middle = choosen index to compare with
NNFeature(model, y_test, x_test, choosen_index, num_classes, input_shape) #third = choosen index to compare with

