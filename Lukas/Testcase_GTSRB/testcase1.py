import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import load_model

def cnn(x_train, y_train, x_test, y_test, batch_size, epochs, num_classes, input_shape, result_file):

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                                                  mode='auto')
    model = load_model('testcaseLukas_model')
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #             epochs=epochs,
    #             verbose=1,
    #             validation_data=(x_test, y_test),
    #             callbacks=[earlyStopping])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    with open(result_file, "a") as myfile:
        myfile.write("======== Test case 1 ========\n "
                    "\tTest loss: " + str(score[0]) + "\n"
                    "\tTest accuracy:" + str(score[1]) + "\n \n")
        # Close file????

    model.save('testcaseLukas_model')  # creates a HDF5 file 'testcaseLukas_model'