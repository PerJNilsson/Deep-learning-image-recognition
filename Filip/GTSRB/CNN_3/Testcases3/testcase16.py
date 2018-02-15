import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping

# C^2(7) P C^2(5) P C^2(3) P
# FC: 400 200 100 43
# Dropout on layer: FC layers
# Batch normalisation: Conv layers before pooling

def cnn(x_train, y_train, x_test, y_test, batch_size, epochs, num_classes,
        input_shape, drop_out_rate, learn_rate, verbose_value, result_file):


    # Network structure
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(7, 7), strides=(1, 1), padding='same',
                     activation='relu',input_shape=input_shape))
    model.add(Conv2D(8, (7, 7), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(drop_out_rate))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=learn_rate),
                  metrics=['accuracy'])

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                                                  mode='auto')
    csv_logger = keras.callbacks.CSVLogger('Training/training16.csv', separator=';', append=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose_value,
              validation_split=0.2,
              callbacks=[csv_logger])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    with open(result_file, "a") as myfile:
        myfile.write("======== Test case 16 ========\n "
                    "\tTest loss: " + str(score[0]) + "\n"
                    "\tTest accuracy:" + str(score[1]) + "\n \n")
        # Close file????

