# Part of the code in readTrafficSigns and readTrafficTestSigns is code copied from the GTSRB-webpage

import matplotlib.pyplot as plt
import csv
import numpy as np
from keras.utils import np_utils
from PIL import Image
from PIL import ImageOps
import copy
import random



# Loads traffic signs from GTSRB dataset.
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

    print('readTrafficTrainingSigns=Done')
    return images, labels, image_array, hist_labels

# This fucntion will create pictures with 1/4 of the picture zero padded. If the class has to few images it will create
# four more images (1/4 taken away from all corner). If the class has a a decent amout, 2 crops will be generated in
# right upper and left bottom corner. If it has "too many" images it will only create one zero padded corner, randomly chosen.
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

# Function to check which classes are under- or over represented.
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

# Reads the test signs for the GTSRB dataset.
def readTrafficTestSigns(rootpath, crop, size):
    images = [] # images
    labels = [] # corresponding labels
    image_array = []
    hist_labels = []
    prefix = './GTSRB/Final_Test/Images' + '/'
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

# FUunction to shu
def shuffle(labels, image_array):
    #s_images = []
    s_labels = []
    s_image_array =[]
    shuffled_numbers = np.random.permutation(len(labels))
    for number in shuffled_numbers:
        #s_images.append(images[number])
        s_labels.append(labels[number])
        s_image_array.append(image_array[number])
    s_labels = np.array(s_labels)
    #s_image_array = np.array(s_image_array).astype(np.dtype(float))
    print('Shuffle = Done')
    return s_labels, s_image_array

def getHistogram(labels):
    labels = list(map(int, labels))
    plt.hist(labels, bins=43)
    plt.show()


##########################Running the functions########################################

crop = 'y'  # To crop the data 'y' for yes 'n' for no
size = 32   # Size of the resized picture size*size pixels
colormode = 'RGB' # string: RGB, P -palette mode(only uses a small number of colors), or L - grayscale
rootpath = './GTSRB/Final_Training/Images'
test_rootpath = './GTSRB/Final_Test/Images'
images, labels, image_array, hist_labels = readTrafficSigns(rootpath, crop, size, colormode)
# Need to change tensor for palette and greyscale, diffenrent amount of color channels.

labels2 = copy.copy(labels)
labels2 = np.array(labels2)
image_array2 = copy.copy(image_array)

min_images = 800
max_images = 1300
list_to_few, list_to_many = smoothDistribution(labels, min_images, max_images)
plt.title('Before image manipulation')
plt.xlabel('Class')
plt.ylabel('Number of images')
#getHistogram(labels)

final_labels = labels
final_image_array = image_array
image_array1, labels1 = manipulateImages(image_array, labels, size, list_to_few, list_to_many)
final_labels = np.concatenate((labels1, labels2))
final_image_array = np.concatenate((image_array1, image_array2))
plt.title('After image manipulation')
plt.xlabel('Class')
plt.ylabel('Number of images')
#getHistogram(final_labels)

#plt.title('Test image')
#plt.imshow(final_image_array[11])
#plt.show()
s_labels, s_image_array = shuffle(final_labels, final_image_array)

s_labels = np.array(s_labels)
s_labels = np_utils.to_categorical(s_labels, 43) # Makes binary class-matrix
print('Total number of images:', len(s_labels))


s_image_array = np.array(s_image_array).astype(np.dtype(float))
s_image_array /= 255



[test_images, test_labels, test_image_array, test_hist_label] = readTrafficTestSigns(rootpath, crop, size)

#s_image_array = s_image_array[1:4000]
#s_labels = s_labels[1:4000]
#test_image_array = test_image_array[1:400]
#test_labels = test_labels[1:400]


########## KERAS CODE ##################

input_dim = (size, size, 3)
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


batch_size = 6
epochs =1
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)
datagen.fit(s_image_array)

model = Sequential()

model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_dim, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(BatchNormalization(axis=1))

# Add fully connected layer
model.add(Flatten())  # Making the eights from convL 1-dim
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization(axis=1))
model.add(Dense(1024, activation='relu'))
# Add output layer
model.add(Dropout(0.3))
#model.add(BatchNormalization(axis=1))



model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Change verbose to see progress

history = model.fit(s_image_array, s_labels, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.3)



score_training = model.evaluate(s_image_array, s_labels, verbose=1)
score = model.evaluate(test_image_array, test_labels, verbose=1)

print('For the TRAINING SET:')
print('Percentage of images recognized: %s' %score_training[1])
print('Energy function: %s' %score_training[0])


print('For the TEST SET:')
print('Percentage of images recognized: %s' %score[1])
print('Energy function: %s' %score[0])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['trainloss', 'vallos'], loc='upper left')
plt.show()

print('###################################################DONE#############################################')