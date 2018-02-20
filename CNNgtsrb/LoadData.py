# Part of the code in readTrafficSigns and readTrafficTestSigns is code copied from the GTSRB-webpage

import matplotlib.pyplot as plt
import csv
import numpy as np
from keras.utils import np_utils
from PIL import Image
from PIL import ImageOps
import copy
import random



def readTrafficSigns(rootpath, crop, size, colormode):
    images = [] # images
    labels = [] # corresponding labels
    image_array = []
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            if crop == 'y':
                tmp_im = Image.open(prefix + row[0]).crop((int(row[3]), int(row[4]), int(row[5]), int(row[6])))
                tmp_im = tmp_im.convert(colormode)
                images.append(tmp_im.resize((size, size), Image.ANTIALIAS)) #Image: ANTIALIAS, BICUBIC,BILINEAR, NEAREST
            elif crop == 'n':
                tmp_im = Image.open(prefix + row[0])
                tmp_im = tmp_im.convert(colormode)
                images.append(tmp_im.resize((size, size), Image.ANTIALIAS))
            else:
                print('Wrong argument: \'y\' for crop, \'n\' for original picture')
                break
            labels.append(row[7]) # the 8th column is the label
            image_array.append(np.asarray(images[-1]))
        print('Training images progress:', c+1, '/43')
        gtFile.close()
    hist_labels = labels
    image_array = np.array(image_array).astype(np.dtype(float))
    image_array /= 255
    print('readTrafficTrainingSigns=Done')
    return images, labels, image_array, hist_labels


def manipulateImages(image_array, labels, size, list_to_few, list_to_many):
    half_size = int(size/2)
    n_images = len(labels)
    d_image_array = []
    d_labels = []
    # for nIt in range(0,43):
    #     class_obj = str(nIt)
    #     indices = [i for i, x in enumerate(labels) if x == class_obj]
    #     if len(indices) < 500:
    #         print('Need to make more data on class', nIt)
    for nIt in range(0, n_images):
        tmp_im1 = image_array[nIt]
        tmp_im2 = copy.copy(tmp_im1)
        tmp_label = int(labels[nIt])
        tmp_im3 = copy.copy(tmp_im1)
        tmp_im4 = copy.copy(tmp_im1)
        if tmp_label in list_to_few:
            for k in range(0, half_size):
                for l in range(half_size, size):
                    tmp_im1[k][l] = [0, 0, 0]  # Set pixel to black
            d_image_array.append(tmp_im1)
            d_labels.append(tmp_label)
            for i in range(half_size, size):
                for j in range(0, half_size):
                    tmp_im2[i][j] = [0, 0, 0]  # Set pixel to black
            d_image_array.append(tmp_im2)
            d_labels.append(tmp_label)
            for k in range(0, half_size):
                for l in range(0, half_size):
                    tmp_im3[k][l] = [0, 0, 0]  # Set pixel to black
            d_image_array.append(tmp_im3)
            d_labels.append(tmp_label)
            for i in range(half_size, size):
                for j in range(half_size, size):
                    tmp_im4[i][j] = [0, 0, 0]  # Set pixel to black
            d_image_array.append(tmp_im4)
            d_labels.append(tmp_label)

        elif tmp_label in list_to_many:
            r = random.random()
            if r < 0.25:
                for k in range(0, half_size):
                    for l in range(half_size, size):
                        tmp_im1[k][l] = [0, 0, 0]  # Set pixel to black
                d_image_array.append(tmp_im1)
                d_labels.append(tmp_label)
            elif 0.25 < r < 0.5:
                for i in range(half_size, size):
                    for j in range(0, half_size):
                        tmp_im2[i][j] = [0, 0, 0]  # Set pixel to black
                d_image_array.append(tmp_im2)
                d_labels.append(tmp_label)
            elif 0.5 < r < 0.75:
                for k in range(0, half_size):
                    for l in range(0, half_size):
                        tmp_im3[k][l] = [0, 0, 0]  # Set pixel to black
                d_image_array.append(tmp_im3)
                d_labels.append(tmp_label)
            else:
                for i in range(half_size, size):
                    for j in range(half_size, size):
                        tmp_im4[i][j] = [0, 0, 0]  # Set pixel to black
                d_image_array.append(tmp_im4)
                d_labels.append(tmp_label)
        else:
            for k in range(0, half_size):
                for l in range(0, half_size):
                    tmp_im1[k][l] = [0, 0, 0]  # Set pixel to black
            d_image_array.append(tmp_im1)
            d_labels.append(tmp_label)
            for i in range(half_size, size):
                for j in range(half_size, size):
                    tmp_im2[i][j] = [0, 0, 0]  # Set pixel to black
            d_image_array.append(tmp_im2)
            d_labels.append(tmp_label)
    return d_image_array, d_labels

def smoothDistribution(labels, min_images, max_images):
    list_to_few = []
    list_to_many = []
    for nIt in range(0, 43):
        class_obj = str(nIt)
        indices = [i for i, x in enumerate(labels) if x == class_obj]
        if len(indices) < min_images:
            list_to_few.append(nIt)
        if len(indices) > max_images:
            list_to_many.append(nIt)
    return list_to_few, list_to_many

def readTrafficTestSigns(rootpath, crop, size):
    images = [] # images
    labels = [] # corresponding labels
    image_array = []
    hist_labels = []
    prefix = rootpath + '/'
    gtTestFile = open(prefix + 'GT-final_test.csv')
    gtTestReader = csv.reader(gtTestFile, delimiter=';')
    next(gtTestReader)
    nIt = 1
    for row in gtTestReader:
        if crop == 'y':
            tmp_im = Image.open(prefix + row[0]).crop((int(row[3]), int(row[4]), int(row[5]), int(row[6])))
            images.append(tmp_im.resize((size, size), Image.ANTIALIAS))
        elif crop == 'n':
            images.append(Image.open(prefix + row[0]).resize((size, size), Image.ANTIALIAS))
        else:
            print('Wrong argument: \'y\' for crop, \'n\' for original picture')
            break
        if nIt % 1000 == 0:
            print('Test images progress:', nIt, '/12000')
        nIt = nIt+1
        labels.append(row[7])  # the 8th column is the label
        hist_labels.append(int(row[7]))
        image_array.append(np.asarray(images[-1]))
    gtTestFile.close()
    hist_labels = labels
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels, 43)  # Makes binary class-matrix
    image_array = np.array(image_array).astype(np.dtype(float))
    image_array /= 255
    print('readTrafficTestSigns=Done')
    return images, labels, image_array, hist_labels


def shuffle(images, labels, image_array):
    s_images = []
    s_labels = []
    s_image_array =[]
    shuffled_numbers = np.random.permutation(len(images))
    for number in shuffled_numbers:
        s_images.append(images[number])
        s_labels.append(labels[number])
        s_image_array.append(image_array[number])
    s_image_array = np.array(s_image_array).astype(np.dtype(float))
    s_image_array /= 255
    s_labels = np.array(s_labels)
    print('Shuffle = Done')
    return s_images, s_labels, s_image_array

def getHistogram(labels):
    labels = list(map(int, labels))
    plt.hist(labels, bins=43)
    plt.show()



# To do: Zero padding

##########################TEST PART OF CODE########################################

crop = 'y'
size = 16
colormode = 'RGB' # string: RGB, P -palette mode(only uses a small number of colors), or L - grayscale
rootpath = './GTSRB/Final_Training/Images'
test_rootpath = './GTSRB/Final_Test/Images'
images, labels, image_array, hist_labels = readTrafficSigns(rootpath, crop, size, colormode)
#test_images, test_labels, test_image_array, test_hist_labels = readTrafficTestSigns(test_rootpath, crop, size)


# plt.imshow(image_array[0])
# plt.show()
# print(len(labels))
min_images = 1000
max_images = 1400
list_to_few, list_to_many = smoothDistribution(labels, min_images, max_images)
getHistogram(labels)
image_array, labels = manipulateImages(image_array, labels, size, list_to_few, list_to_many)
getHistogram(labels)



#s_images, s_labels, s_image_array = shuffle(images, labels, image_array)

# labels = np.array(labels)
# labels = np_utils.to_categorical(labels, 43) # Makes binary class-matrix

#divideImages(images, hist_labels)
#plt.imshow(image_array[0])
#plt.show()
#getHistogram(hist_labels)
#s_image_array = s_image_array.reshape(s_image_array.shape[0], 32, 32, 3)



print('###################################################DONE#############################################')