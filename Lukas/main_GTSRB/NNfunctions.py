from __future__ import print_function
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

def NNPixel(x_test,index,number_of_neighbors):
    choosen_img = x_test[index]
    plt.imshow(choosen_img)
    plt.show()
    loneerror = ltwoerror = np.zeros(len(x_test))   # L1 and L2 error vector
    for i in range(len(x_test)):
        loneerror[i] = np.sum(np.abs(x_test[i] - choosen_img))  # L1 error vector
        ltwoerror[i] = np.sum(np.square(x_test[i] - choosen_img))  # L2 error vector

    index_min_lone = index_min_ltwo = np.zeros(number_of_neighbors).astype(int)
    for k in range(number_of_neighbors):
        index_min_lone[k] = loneerror.argmin()
        index_min_ltwo[k] = ltwoerror.argmin()
        loneerror[index_min_lone[k]] = np.power(10,20)
        ltwoerror[index_min_lone[k]] = np.power(10,20)

    print(index_min_lone)
    print(index_min_ltwo)
    for p in range(0,number_of_neighbors):
        plt.imshow(x_test[index_min_lone[p]])
        plt.show()

def NNFeature(model, y_test, x_test,index, num_classes):
    print('TODO')
