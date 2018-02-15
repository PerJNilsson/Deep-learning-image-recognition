
import csv
from PIL import Image
from numpy import array
from numpy import asarray
from numpy import clip
import numpy as np
import matplotlib.pyplot as plt

def readData(path, classRange, imSize):

    images = []
    arr = []
    labels = []
    i = 1
    for c in range(0, classRange):
        prefix = path + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file

        for row in gtReader:
            if i % 1000 == 0:
                print('Class: ', c, 'Image', i)
            im = Image.open(prefix + row[0])
            images.append(im.crop((int(row[3]), int(row[4]), int(row[5]), int(row[6]))).resize(imSize))
            labels.append(int(row[7]))  # the 8th column is the label
            arr.append(array(images[-1]))
            i = i+1
        gtFile.close()
    return arr, labels, images


def readValidationData(path, imSize, num_files):

    images = []
    arr = []
    labels = []
    i = 1

    prefix = path + '/'
    gtFile = open(prefix + 'GT-final_test_with_classes.csv')  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    next(gtReader)  # skip header

    for row in gtReader:
        if i > num_files:
            break
        if i % 1000 == 0:
            print('Image', i)
        im = Image.open(prefix + row[0])
        images.append(im.crop((int(row[3]), int(row[4]), int(row[5]), int(row[6]))).resize(imSize))
        labels.append(int(row[7]))  # the 8th column is the label
        arr.append(array(images[-1]))
        i = i+1

    gtFile.close()
    return arr, labels, images

def oneHotEncode(l, numClasses):

    onehot_encoded = list()
    for value in l:
        vector = [0 for _ in range(numClasses)]
        if value < numClasses:
            vector[value] = 1
        onehot_encoded.append(vector)

    return asarray(onehot_encoded)

# Takes array of images with pixel values (0,1)
def normaliseImage(arr):
    num_images = arr.shape[0]
    x_dim = arr.shape[1]
    y_dim = arr.shape[2]
    channels = arr.shape[3]
    arr2 = []
    for i in range(0,num_images):
        temp_arr = []
        for j in range(0,channels):
            a = arr[i,:,:,j]
            m = a.mean()
            v = a.std()
            temp_arr.append(clip((a-m)/v+0.5,0,1))

        temp_arr = np.rollaxis(asarray(temp_arr),0,3)
        arr2.append(temp_arr)

    return  asarray(arr2)






