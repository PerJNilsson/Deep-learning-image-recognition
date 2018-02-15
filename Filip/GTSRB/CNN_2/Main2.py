

from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from readData import readData
from readData import readValidationData
from readData import oneHotEncode
from readData import normaliseImage

import Testcases2.testcase1
import Testcases2.testcase2
import Testcases2.testcase3
import Testcases2.testcase4
import Testcases2.testcase5
import Testcases2.testcase6
import Testcases2.testcase7
import Testcases2.testcase8
import Testcases2.testcase9
import Testcases2.testcase10
import Testcases2.testcase11
import Testcases2.testcase12
import Testcases2.testcase13
import Testcases2.testcase14
import Testcases2.testcase15
import Testcases2.testcase16
import Testcases2.testcase17
import Testcases2.testcase18
import Testcases2.testcase19


# NOTES: split training into validation data also. Shuffle each training session. (Is constant throughout the epochs)
# Look at validation accuracy instead of test accuracy
# - add drop out
# Plot
# Batch normalisation after layers -> Gaussian
# Data augmentation
# Confusion matrix


epochs = 1 #30
test_size = 50 #15000
training_size = 500 # 40000
num_classes = 43 #43
result_file = "test_run_results.txt"

# input image dimensions
img_x, img_y = 32, 32

# load data sets
#arr, labels, images = readData('C:/Users/Filip/Documents/Kandidat/GTSRB/Final_Training/Images', num_classes, (img_x, img_y))

#v_arr, v_labels, v_images = readValidationData('C:/Users/Filip/Documents/Kandidat/GTSRB/Final_Test/Images',
                                               #(img_x, img_y), test_size)
#x_train = np.asarray(arr)
#x_test = np.asarray(v_arr)

#np.save("xtrain", arr)
#np.save("ytrain", labels)
#np.save("xtest", v_arr)
#np.save("ytest", v_labels)

x_train = np.load("xtrain.npy")
labels = np.load("ytrain.npy")
x_test = np.load("xtest.npy")
v_labels = np.load("ytest.npy")

print(x_train.shape)

randomIndexMatrix = np.arange(x_train.shape[0])
np.random.shuffle(randomIndexMatrix)

x_train = x_train[randomIndexMatrix]

labels = labels[randomIndexMatrix]
y_train = oneHotEncode(labels, num_classes)

y_test = oneHotEncode(v_labels, num_classes)

x_train = x_train[:training_size]
y_train = y_train[:training_size]
x_test = x_test[:test_size]
y_test = y_test[:test_size]

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 3)


# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train_processed = normaliseImage(x_train)
x_test_processed = normaliseImage(x_test)


print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('============================')

print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)

print('y_train shape', y_train.shape)
print('y_test shape', y_test.shape)

# Visualize data
#plt.imshow(x_train[0])
#plt.show()






batch_size = 10
lr = 0.01

train_data = (x_train, x_train_processed)
test_data = (x_test, x_test_processed)

drop_outs = (0.2, 0.4, 0.6)
myfile = open(result_file, "w")
for j in range(0, 2):
    x1 = train_data[j]
    x2 = test_data[j]
    for i in range(0,3):
        with open(result_file, "a") as myfile:
            myfile.write("======================================\n"
                         "============= TEST RUN " + str(j+1) + ", " + str(i+1) + " =============\n"
                         "======================================\n\n")
        myfile.close()

        do = drop_outs[i]

        Testcases2.testcase1.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase2.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase3.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase4.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase5.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase6.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase7.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase8.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase9.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase10.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase11.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase12.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase13.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase14.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase15.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase16.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase17.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase18.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
        Testcases2.testcase19.cnn(x1, y_train, x2, y_test, batch_size, epochs, num_classes, input_shape, do, lr,
                                 result_file)
with open(result_file, "a") as myfile:
    myfile.write("======================================\n"
                 "======= TEST ENDED SUCCESSFULLY ======\n"
                 "======================================\n\n")
#file = open("results.txt", "w")
#file.write()
#file.close()