
# 3. Import libraries and modules
import numpy as np

np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Print a random image from mnist
#print (X_train.shape)
#plt.imshow(X_train[346])
#plt.show()

# Preprocess training input data
# We only have a depth of 1 (RBG = 3). Need to specify this
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)


#print (X_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# print (y_train[:10])
# The y_train and y_test data are not split into 10 distinct class labels, but rather are represented as a single
# # array with the class values. The fix:
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# If one wanna do faster computations, just to test the network
#Y_train = Y_train[:10000]
#X_train = X_train[:10000]
#print (Y_train.shape)


# Declare sequential model
model = Sequential()

#  Add the first (input) layer
#  32 convL, 3x3 matrices. Output after this layer would be a 32x28x28 matrice
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28),
                        dim_ordering='th'))


model.add(Convolution2D(32, 3, 3, activation='tanh')) # Add second layer
model.add(Convolution2D(32, 3, 3, activation='tanh')) # Add third layer
model.add(Convolution2D(32, 3, 3, activation='relu')) # # Add forth  layer
model.add(MaxPooling2D(pool_size=(2,2)))  # 5th: MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the previous layer and taking the max of the 4 values in the 2x2 filter.
model.add(Dropout(0.25)) # Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

# Add fully connected layer
model.add(Flatten())  # Making the eights from convL 1-dim
model.add(Dense(128, activation='relu'))

# Add output layer
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Defining things..
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Change verbose to see progress

model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)


print ('Percentage of images recognized: %s' %score[1])
