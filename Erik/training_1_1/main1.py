from keras import backend as K

from case1 import case1
from case2 import case2
from case3 import case3
from case4 import case4
from case5 import case5
from case6 import case6
from case7 import case7
from case8 import case8

training_name = 'training_1_1'
training_path = 'Data/Final_Training/Images' # make sure this is the path to training set

K.set_image_data_format('channels_first')

case1(training_path, training_name)
case2(training_path, training_name)
case3(training_path, training_name)
case4(training_path, training_name)
case5(training_path, training_name)
case6(training_path, training_name)
case7(training_path, training_name)
case8(training_path, training_name)

