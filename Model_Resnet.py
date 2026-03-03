import numpy as np
import cv2 as cv
from keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from Evaluation import evaluation


def Model_RESNET(train_data, train_target, test_data, test_target, Batch_size):

    Classes = test_target.shape[-1]
    inputs = (32, 32, 3)

    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=inputs))
    base_model.add(Dense(units=Classes, activation='linear'))  # units=train_target.shape[1]
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    base_model.fit(Train_X, train_target, epochs=150, batch_size=Batch_size, validation_data=(Test_X, test_target))
    pred = base_model.predict(Test_X)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred
