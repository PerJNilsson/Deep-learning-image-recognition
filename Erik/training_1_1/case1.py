from helper_func import conv_net, training, lr_schedule, get_class, basic_preprocess

from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.optimizers import SGD
import os

def preprocess(img):
    return basic_preprocess(img)

def import_training_imgs(path):
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(path, '*/*.ppm'))
    np.random.shuffle(all_img_paths)

    for img_path in all_img_paths:
        img = preprocess(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

    x = np.array(imgs, dtype='float32')
    y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
    return x, y

def case1():
    name = 'case1'
    model = conv_net()
    sgd = SGD(lr=lr)  # Momentum and decay not active as is. Could be changed later to improve performance
    model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    X, Y = import_training_imgs()
    training()