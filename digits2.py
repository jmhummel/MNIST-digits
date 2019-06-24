import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

np.random.seed(123)  # for reproducibility

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])
plt.show()

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# 7. Define model architecture
model = Sequential()
model.add(Flatten(input_shape=(1, 28, 28)))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))

"""
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
"""

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
model.fit(X_train, Y_train, batch_size=32, nb_epoch=40, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
# [0.1453707801291719, 0.9568]

w1 = model.get_layer(index=1).get_weights()[0]
w1 = np.squeeze(w1)
w1 = w1.reshape((28, 28, 25))
fig, axs = plt.subplots(5, 5, figsize=(8, 8))
fig.subplots_adjust(hspace=.5, wspace=.001)
axs = axs.ravel()
for i in range(25):
    axs[i].imshow(w1[:, :, i])
    axs[i].set_title(str(i))
plt.show()

w2 = model.get_layer(index=2).get_weights()[0]
w2 = np.squeeze(w2)
w2 = w2.reshape((5, 5, 10))
fig, axs = plt.subplots(5, 2, figsize=(8, 8))
fig.subplots_adjust(hspace=.5, wspace=.001)
axs = axs.ravel()
for i in range(10):
    axs[i].imshow(w2[:, :, i])
    axs[i].set_title(str(i))
plt.show()

w3 = w2.reshape((25, 10))
w4 = np.zeros((28, 28, 10))
fig, axs = plt.subplots(5, 2, figsize=(8, 8))
fig.subplots_adjust(hspace=.5, wspace=.001)
axs = axs.ravel()

for i in range(10):
    for j in range(25):
        w4[:, :, i] += w1[:, :, j] * w3[j, i]
    axs[i].imshow(w4[:, :, i])
    axs[i].set_title(str(i))
plt.show()
