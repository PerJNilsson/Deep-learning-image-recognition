import glob
import os

import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from skimage import transform, io

IMG_SIZE = 48
NUM_CLASSES = 43


# Preprocessing with only crop and standard size
def preprocess_img(img):
    # Central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2: centre[0] + min_side // 2,
          centre[1] - min_side // 2: centre[1] + min_side // 2,
          :]

    # Rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # Roll color axis to 0
    img = np.rollaxis(img, -1)

    return img


# Returns the class by splitting the path - folder identifies class.
def get_class(img_path):
    return int(img_path.split('/')[-2])


# Function that defines the structure of the network
def conv_net():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


# Governs decay of learning rate.
def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


# Imports and shuffles the images. Will only work with the path to the GTSRB Training directory.
def import_training_imgs(path):
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(path, '*/*.ppm'))
    np.random.shuffle(all_img_paths)

    for img_path in all_img_paths:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

    x = np.array(imgs, dtype='float32')
    y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
    return x, y


# Main program
training_path = 'path_to_training_set'
# Parameters for training
lr = 0.01
batch_size = 32
epochs = 30

K.set_image_data_format('channels_first')
model = conv_net()

# Let's train the model using SGD + momentum

sgd = SGD(lr=lr)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X, Y = import_training_imgs(training_path)
# training
model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                     ModelCheckpoint('Trained_models/ConvNet_1_1.h5', save_best_only=True)])
