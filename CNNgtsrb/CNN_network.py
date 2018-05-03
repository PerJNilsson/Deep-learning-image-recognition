def CNN_network(test_image_array, test_labels, s_image_array, s_labels, input_dim, batch_size, epochs):

    import keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt

    batch_size = 6
    epochs =1
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    datagen.fit(s_image_array)

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_dim, padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=1))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=1))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    model.add(BatchNormalization(axis=1))

    # Add fully connected layer
    model.add(Flatten())  # Making the eights from convL 1-dim
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(1024, activation='relu'))
    # Add output layer
    model.add(Dropout(0.3))
    #model.add(BatchNormalization(axis=1))



    model.add(Dense(43, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Change verbose to see progress

    history = model.fit(s_image_array, s_labels, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.3)



    score_training = model.evaluate(s_image_array, s_labels, verbose=1)
    score = model.evaluate(test_image_array, test_labels, verbose=1)

    print('For the TRAINING SET:')
    print('Percentage of images recognized: %s' %score_training[1])
    print('Energy function: %s' %score_training[0])


    print('For the TEST SET:')
    print('Percentage of images recognized: %s' %score[1])
    print('Energy function: %s' %score[0])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['trainloss', 'vallos'], loc='upper left')
    plt.show()
