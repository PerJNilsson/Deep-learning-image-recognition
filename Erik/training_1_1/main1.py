from keras import backend as K

from case1 import case1

K.set_image_data_format('channels_first')

case1()