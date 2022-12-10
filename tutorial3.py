import pandas as pd
import numpy as np
import sklearn
import keras
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist
(X_train, Y_train), (X_test, Y_test) = \
mnist.load_data(path="mnist.npz")

#for idx in range(24):
# plt.subplot(4,6,idx+1)
# plt.imshow(X_train[idx],cmap='gray')
# plt.axis('off')
# plt.title(str(Y_train[idx]))
#plt.tight_layout()

Y_train = X_train
Y_test = X_test

# Normalize the images
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Subtract the smallest pixel value from the image
X_train -= np.min(X_train)
X_test -= np.min(X_test)

# Divide the image by the largest pixel value after subtraction
X_train /= np.max(X_train)
X_test /= np.max(X_test)

X_test=X_test.reshape(-1,28,28,1)
X_train=X_train.reshape(-1,28,28,1)

# Print the shape of the training and test data
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Define convolutional autoencoder
c = 30
cnn = keras.models.Sequential()
cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3),
 strides=(2,2),
 activation='relu',
input_shape=X_train.shape[1:]))
cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3),
 activation='relu'))
cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3),
 strides=(2,2),
 activation='relu'))
cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3),
 activation='relu'))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(100, activation='sigmoid'))
cnn.add(keras.layers.Dense(c, activation='sigmoid'))
cnn.add(keras.layers.Dense(7*7*32, activation='sigmoid'))
cnn.add(keras.layers.Reshape((7,7,32)))
cnn.add(keras.layers.Conv2DTranspose(32, (3, 3),
 activation='relu',
padding='same' ,
 strides=(2, 2)))
cnn.add(keras.layers.Conv2DTranspose(64, (3, 3),
 activation='relu',
padding='same' ,strides=(2, 2)))
cnn.add(keras.layers.Conv2DTranspose(1, (3, 3),
 activation='relu',
padding='same' ))
cnn.compile(loss='mse', optimizer='adam', metrics=['mse'])

#Model training
cnn.fit(X_train, X_train, epochs=5, batch_size=128, validation_data=(X_test, X_test))
score = cnn.evaluate(X_test, X_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Visualize the reconstructed images
X_test_pred = cnn.predict(X_test)

# Generating noise samples
X_trainN = np.random.poisson(X_train)
X_testN = np.random.poisson(X_test)
   
# Training the model
daa = cnn
history = daa.fit(X_trainN, Y_train, validation_data = (X_testN,Y_test), epochs=5, batch_size=128)
    
# Splitting into two models again
def extract_layers(daa, starting_layer_ix, ending_layer_ix):
 # create an empty model
 daa_new = keras.Sequential()
 for ix in range(starting_layer_ix, ending_layer_ix + 1):
     curr_layer = daa.get_layer(index=ix)
     # copy this layer over to the new model
     daa_new.add(curr_layer)
 return daa_new
encode_layer = extract_layers(daa,0,6)
decode_layer = extract_layers(daa,7,11)

# Original output
for i in np.arange(5):
 plt.figure(1)
 plt.subplot(1,5,i+1)
 plt.imshow(X_train[i].reshape(28,28),cmap='gray')
 plt.axis('off')
plt.title('Original sample')
plt.tight_layout()

# Noisy output
for i in np.arange(5):
 plt.figure(2)
 plt.subplot(1,5,i+1)
 plt.imshow(X_trainN[i].reshape(28,28),cmap='gray')
 plt.axis('off')
plt.title('Noisy sample')
plt.tight_layout()

# Denoised output
for i in np.arange(5):
 plt.figure(3)
 encode_out = encode_layer.predict(X_trainN.reshape(-1,28,28,1))
 decode_out = decode_layer.predict(encode_out)
 plt.subplot(1,5,i+1)
 plt.imshow(decode_out[i].reshape(28,28),cmap='gray')
 plt.axis('off')
plt.title('Denoised prediction')
plt.tight_layout()

