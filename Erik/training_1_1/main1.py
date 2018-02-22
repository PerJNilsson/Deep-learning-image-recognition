from keras import backend as K

from case1 import case1

training_name = 'training_1_1'
training_path = 'Data/Final_Training/Images' # make sure this is the path to training set

K.set_image_data_format('channels_first')

case1(training_path, training_name)
