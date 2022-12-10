import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
import keras
from matplotlib import pyplot as plt

X = np.load("X_mfcc_10000.npy")
Y = np.load("Y_mfcc_10000.npy")

X = np.moveaxis(X, [0, 1, 2], [0, 2, 1])

def MyScaler(X):
 for i, x in enumerate(X):
     X[i,:,:]=X[i,:,:]/np.std(x)
 return X
X = MyScaler(X)
X_train1, X_test1, Y_train1, Y_test1 = sklearn.model_selection.train_test_split(X,Y,test_size=0.1) 
X_train2, X_test2, Y_train2, Y_test2 = sklearn.model_selection.train_test_split(X,Y,test_size=0.1) 
X_train3, X_test3, Y_train3, Y_test3 = sklearn.model_selection.train_test_split(X,Y,test_size=0.1) 
X_train4, X_test4, Y_train4, Y_test4 = sklearn.model_selection.train_test_split(X,Y,test_size=0.1) 

model1 = keras.Sequential()
model1.add(keras.layers.LSTM(16))
model1.add(keras.layers.Dense(2, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history1 = model1.fit(X_train1, Y_train1, validation_data = (X_test1,Y_test1), epochs=5, batch_size=64)
model1.summary()

model2 = keras.Sequential()
model2.add(keras.layers.LSTM(16, return_sequences=True))
model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dense(2, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model2.fit(X_train2, Y_train2, validation_data = (X_test2,Y_test2), epochs=5, batch_size=64)
model2.summary()

model3 = keras.Sequential()
model3.add(keras.layers.Flatten(input_shape=(126, 40)))
model3.add(keras.layers.Dense(32, activation='relu'))
model3.add(keras.layers.Dense(2, activation='relu'))
model3.add(keras.layers.Dense(2,activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history3 = model3.fit(X_train3, Y_train3, validation_data = (X_test3,Y_test3), epochs=5, batch_size=64)
model3.summary()

model4 = keras.Sequential()
model4.add(keras.layers.LSTM(26))
model4.add(keras.layers.Dense(16, activation='relu'))
model4.add(keras.layers.Dense(2, activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history4 = model4.fit(X_train1, Y_train1, validation_data = (X_test1,Y_test1), epochs=5, batch_size=64)
model4.summary()