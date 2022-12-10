import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
import keras
from matplotlib import pyplot as plt

X = np.load("X_mfcc_10000.npy")
Y = np.load("Y_mfcc_10000.npy")

X = np.moveaxis(X, [0, 1, 2], [0, 2, 1])

def min_max_scaler(data):
    return (data - data.min()) / (data.max() - data.min())

X = min_max_scaler(X)
Y = min_max_scaler(Y)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

input1 = keras.Input(shape=(126,40))
input2 = keras.Input(shape=(126,40,1))

x1 = input1
x1 = keras.layers.LSTM(16, return_sequences=True)(x1)
x1 = keras.layers.Flatten()(x1)
  
x2 = input2
x2 = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(x2)
x2 = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(x2)
x2 = keras.layers.MaxPooling2D(pool_size=(2,2))(x2)
x2 = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(x2)
x2 = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(x2)
x2 = keras.layers.MaxPooling2D(pool_size=(2,2))(x2)
x2 = keras.layers.Flatten()(x2)
x2 = keras.layers.Dense(100, activation='sigmoid')(x2)

#Concatenating

z = keras.layers.Concatenate()([x1,x2])
z = keras.layers.Dense(2, activation='softmax')(z)
model3 = keras.Model(inputs=[input1, input2], outputs=z)
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history3 = model3.fit([X_train, X_train], Y_train, validation_data = ([X_test, X_test],Y_test), epochs=5, batch_size=64)