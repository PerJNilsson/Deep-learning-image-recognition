import glob
import os

import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, EarlyStopping
from keras import backend as K
from skimage import transform, io

IMG_SIZE = 48
NUM_CLASSES = 43

# Preprocessing with only crop and standard size
def basic_preprocess(img):
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
    lr = 0.01
    return lr * (0.1 ** int(epoch / 10))

# training
def training():
    lr_scheduler = LearningRateScheduler(lr_schedule)
    model_checkpoint = ModelCheckpoint(os.path.join('Trained_models/' + TRAINING_NAME, name + '.h5'),
                                       save_best_only=True)
    csv_logger = CSVLogger(os.path.join('Logs/' + TRAINING_NAME, name + '.csv'), separator=';')
    early_stoppping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    callbacks = [lr_scheduler, model_checkpoint, csv_logger, early_stoppping]

    model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=callbacks)
