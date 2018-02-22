from helper_func import conv_net, training, basic_preprocess, import_training_imgs
from helper_func import norm_s, norm_v
from keras.optimizers import SGD
import numpy as np

NUM_CLASSES = 43
lr = 0.01
case_name = 'case7'

def preprocess(img):
    img = basic_preprocess(img)
    img = norm_s(img)
    img = norm_v(img)
    # Roll color axis to 0
    img = np.rollaxis(img, -1)
    return img

def case7(training_path, training_name):
    model = conv_net()
    sgd = SGD(lr=lr)  # Momentum and decay not active as is. Could be changed later to improve performance
    model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    X, Y = import_training_imgs(training_path, preprocess)
    training(model, X, Y, training_name, case_name)