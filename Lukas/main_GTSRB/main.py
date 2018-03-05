from __future__ import print_function

import numpy as np

from Testcase_GTSRB import testcase1
from Testcase_GTSRB import testcaseLukas
from keras.models import load_model
from keras.utils import plot_model

# NOTES: split training into validation data also. Shuffle each training session. (Is constant throughout the epochs)
# Look at validation accuracy instead of test accuracy
# - add drop out
# Plot
# Batch normalisation after layers -> Gaussian
# Data augmentation
# Confusion matrix

batch_size = 100
epochs = 1
validation_size = 100
num_classes = 10
result_file = "test_run_results.txt"

# input image dimensions
img_x, img_y = 32, 32

# load data sets
arr, labels, images = readData('C:\\Users\\nystr\\GTSRB\\Final_Training\\Images', num_classes, (img_x, img_y))

v_arr, v_labels, v_images = readValidationData('C:\\Users\\nystr\\GTSRB\\Final_Test\\Images',
                                               (img_x, img_y), validation_size)

#arr, labels, images = readData('C:/Users/Filip/Documents/Kandidat/GTSRB/Final_Training/Images', num_classes, (img_x, img_y))

#v_arr, v_labels, v_images = readValidationData('C:/Users/Filip/Documents/Kandidat/GTSRB/Final_Test/Images',
         #                                      (img_x, img_y), validation_size)

x_train = np.asarray(arr)
y_train = oneHotEncode(labels, num_classes)

x_test = np.asarray(v_arr)
y_test = oneHotEncode(v_labels, num_classes)

x_test = x_test[:validation_size]
y_test = y_test[:validation_size]

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

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('============================')

print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)

print('y_train shape', y_train.shape)
print('y_test shape', y_test.shape)

print('Testing testcase1!')
myfile = open(result_file, "w")
myfile.write("======================================\n"
             "============= TEST RUN 1 =============\n"
             "======================================\n\n")
myfile.close()

testcase1.cnn(x_train, y_train, x_test, y_test, batch_size, epochs, num_classes, input_shape, result_file)

modelp = load_model('testcaseLukas_model')
print(modelp.summary())
plot_model(modelp, to_file='model.png',show_shapes=True, show_layer_names=True)

with open(result_file, "a") as myfile:
    myfile.write("======================================\n"
                 "======= TEST ENDED SUCCESSFULLY ======\n"
                 "======================================\n\n")
#file = open("results.txt", "w")
#file.write()
#file.close()
