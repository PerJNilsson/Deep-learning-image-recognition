# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
import numpy as np
import PIL
from PIL import Image
from PIL import ImageOps

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrainingSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    x1 = [] # X-coordinate of top-left corner
    y1 = [] # Y-coordinate of top-left corner
    x2 = []  # X-coordinate of bottom-right corner
    y2 = []  # Y-coordinate of bottom-right corner
    # loop over all 42 classes
    for c in range(0,43):
        print('class: ' + str(c))
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
            x1.append(row[3])  # the 3rd column is x1
            y1.append(row[4])  # the 3rd column is y1
            x2.append(row[5])  # the 3rd column is x2
            y2.append(row[6])  # the 3rd column is y2
        gtFile.close()
    return images, labels, x1, y1, x2, y2

def readTestingSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''

    images = []  # images
    labels = []  # corresponding labels
    x1 = [] # X-coordinate of top-left corner
    y1 = [] # Y-coordinate of top-left corner
    x2 = []  # X-coordinate of bottom-right corner
    y2 = []  # Y-coordinate of bottom-right corner
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test.csv')  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    next(gtReader)  # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
        labels.append(row[7])  # the 8th column is the label
        x1.append(row[3])  # the 3rd column is x1
        y1.append(row[4])  # the 3rd column is y1
        x2.append(row[5])  # the 3rd column is x2
        y2.append(row[6])  # the 3rd column is y2
    gtFile.close()
    return images, labels, x1, y1, x2, y2


def traffic_data():
    X_train, Y_train, x1_train, y1_train, x2_train, y2_train = readTrainingSigns('Images')
    X_test, Y_test, x1_test, y1_test, x2_test, y2_test = readTestingSigns('Images_test')

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    x_test = np.array(X_test)
    y_test = np.array(Y_test)

    # x_train = np.load('x_train.npy')
    # x_test = np.load('x_test.npy')
    # y_train = np.load('y_train.npy')
    # y_test = np.load('y_test.npy')

    train_length = len(y_train)
    test_length = len(y_test)

    train_widths = np.zeros(train_length, dtype=int)
    train_heights = np.zeros(train_length, dtype=int)
    test_widths = np.zeros(test_length,dtype=int)
    test_heights = np.zeros(test_length,dtype=int)
    def new_size_train(len_x, len_y, arr):
        x = []
        for i in range(0, len(arr)):
            img = PIL.Image.fromarray(arr[i])
            lim_left =int(x1_train[i])
            lim_upper =int(y1_train[i])
            lim_right = int(x2_train[i])
            lim_lower = int(y2_train[i])

            train_widths[i] = lim_right-lim_left
            train_heights[i] = lim_lower -lim_upper

            img = img.crop((lim_left, lim_upper, lim_right, lim_lower))
            img = img.resize((len_x, len_y))
            img = ImageOps.autocontrast(img, cutoff=0.1) #Can change cutoff ratio
            img = ImageOps.equalize(img)
            x.append(np.array(img))
        return x

    def new_size_test(len_x, len_y, arr):
        x = []
        for i in range(0, len(arr)):
            img = PIL.Image.fromarray(arr[i])
            lim_left =int(x1_test[i])
            lim_upper =int(y1_test[i])
            lim_right = int(x2_test[i])
            lim_lower = int(y2_test[i])

            test_widths[i] = lim_right - lim_left
            test_heights[i] =  lim_lower -lim_upper

            img = img.crop((lim_left, lim_upper, lim_right, lim_lower))
            img = img.resize((len_x, len_y))
            img = ImageOps.autocontrast(img, cutoff=0.1) #Can change cutoff ratio
            img = ImageOps.equalize(img)
            x.append(np.array(img))
        return x

    x_train = new_size_train(48, 48, x_train)
    x_test = new_size_test(48, 48, x_test)
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    np.save('x_train2', x_train)
    np.save('y_train2', y_train)
    np.save('x_test2', x_test)
    np.save('y_test2', y_test)

    return train_heights, train_widths, test_heights, test_widths

train_heights, train_widths, test_heights, test_widths = traffic_data()

print(train_widths)
print(train_heights)
# tot = np.count_nonzero(train_widths)
# print('Percentage with width larger than 100: ' + str((np.count_nonzero(train_widths>100))/tot))
# print('Percentage with height larger than 100: ' + str((np.count_nonzero(train_heights>100))/tot))
# print('Percentage with width smaller than 50: ' + str((np.count_nonzero(train_widths<50))/tot))
# print('Percentage with height smaller than 50: ' + str((np.count_nonzero(train_heights<50))/tot))
# x_train, y_train, x_test, y_test = traffic_data()
# # x_test = np.array(x_test)
# # y_test = np.array(y_test)
# print(x_train.shape)
# print(y_train.shape)







