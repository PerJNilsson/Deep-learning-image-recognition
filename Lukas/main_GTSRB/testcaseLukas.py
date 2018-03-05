import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.models import load_model

def cnn(x_train, y_train, x_test, y_test, batch_size, epochs, num_classes, input_shape, result_file):

    # Network structure
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),padding='same',
                     activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32, (5, 5),padding='same', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False),
                  metrics=['accuracy'])

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                                                  mode='auto')
    #plot_model(model, to_file='model.png')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[earlyStopping])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    with open(result_file, "a") as myfile:
        myfile.write("======== Test case 2 ========\n "
                    "\tTest loss: " + str(score[0]) + "\n"
                    "\tTest accuracy:" + str(score[1]) + "\n \n")
        # Close file????

    model.save('testcaseLukas_model')  # creates a HDF5 file 'testcaseLukas_model'
    del model  # deletes the existing model