seed = 1337  # for reproducibility
import numpy as np
np.random.seed(seed)        # Seed Numpy
import random               # Seed random
random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)    # Seed Tensor Flow

from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform

from dataset import read_MNIST

class Distortion():

    def __init__(self):
        self.rotation_range = 5.0
        self.height_shift_range = 0.1
        self.width_shift_range = 0.1
        self.shear_range = 0.02 * np.pi
        self.zoom_range = [0.95, 1.05]
        self.fill_mode = "nearest"
        self.cval = 1

    # Function definition taken from keras source code
    def apply_affine_distortion(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 0

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                                translation_matrix),
                                         shear_matrix),
                                  zoom_matrix)

        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode=self.fill_mode, cval=self.cval)

        return x


def get_model(input_shape, nb_classes):
    m = Sequential()
    m.add(Dense(800, input_shape=input_shape, init='he_normal'))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(0.5))
    m.add(Dense(800, init='he_normal'))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(0.5))
    m.add(Dense(nb_classes, init='he_normal'))  # Last layer with one output per
    m.add(Activation('softmax'))

    return m

if __name__ == '__main__':
    nb_classes = 10
    ht = wd = 28
    model_file = 'model.h5'
    epochs = 70
    train = 0

    ############# get data ############
    X_train, y_train = read_MNIST('training', 'resources')
    X_test, y_test = read_MNIST('testing', 'resources')

    ## Data normalization
    # mean_image = np.mean(X_train, axis=0)
    # # s_image = np.std(X_train, axis=0)
    # X_train = (X_train - mean_image)
    # X_train = X_train - np.min(X_train) # lowest pt making 0
    # X_train = (2 * X_train / np.max(X_train) - 1.)
    #
    # X_test = (X_test - mean_image)
    # X_test = X_test - np.min(X_test) # lowest pt making 0
    # X_test = (2 * X_test / np.max(X_test) - 1.)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size = 10000, random_state = 42)
    # train-val split
    X_val = X_train[-10000:]
    y_val = y_train[-10000:]
    X_train = X_train[:-10000]
    y_train = y_train[:-10000]

    N = 2
    X = np.empty([X_train.shape[0] * N, X_train.shape[1], X_train.shape[2], X_train.shape[3]])
    Y = np.empty([y_train.shape[0] * N, y_train.shape[1]])
    idx_x = 0
    idx_y = 0
    d = Distortion()
    for i in range(N):
        for j in range(len(X_train)):
            X[idx_x] = d.apply_affine_distortion(X_train[j])
            idx_x += 1
        Y[idx_y: idx_y + len(y_train)] = y_train
        idx_y += len(y_train)
    X_train = X
    y_train = Y

    # Convert to 1-D vector
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)


    if train:
        ########## Model creation ###############
        # Model definition
        input_shape = (ht*wd,)
        m = get_model(input_shape, nb_classes)

        ################### Training #############
        opt = RMSprop()

        # Compile
        m.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
        print(m.summary())

        # Train
        checkpointer = ModelCheckpoint(filepath=model_file,
                                       monitor='val_loss', verbose=1, save_best_only=True)

        hist = m.fit(X_train, y_train,
              batch_size=256, nb_epoch=epochs,
              validation_data=(X_val, y_val), callbacks=[checkpointer])

        # write history
        np.save('history.npy', hist.history)
        training_loss = hist.history['loss']
        training_acc = hist.history['acc']

        plt.figure()
        plt.grid()
        plt.title('Loss vs Epochs')  # summarize history for accuracy
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        legend = ['Training Loss', 'Validation Loss']
        plt.plot(hist.history['loss'], linewidth=1)
        plt.plot(hist.history['val_loss'], linewidth=1)
        plt.legend(legend, loc='best')
        plt.show(block=0)

        plt.figure()
        plt.grid()
        plt.title('Accuracy vs Epochs')  # summarize history for accuracy
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        legend = ['Training Accuracy', 'Validation Accuracy']
        plt.plot(100*np.array(hist.history['acc']), linewidth=1)
        plt.plot(100*np.array(hist.history['val_acc']), linewidth=1)
        plt.legend(legend, loc='best')
        plt.show(block=1)

    ################### Testing #############
    m = load_model(model_file)
    print(m.summary())
    test_result = m.evaluate(X_test, y_test)
    print(test_result)






