import numpy as np
import pandas as pd
import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import tensorflow as tf
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# iris = load_iris()
# X = iris['data']
# Y = iris['target']
# print(iris)
# print(X.shape)
# print(Y.shape)

iris = pd.read_csv('iris.csv')
iris = iris.sample(frac=1)
# print(iris)
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
Y = iris[['Species1', 'Species2', 'Species3']].values
print(X.shape)
print(Y.shape)

# Model for fit:
##################################################################################################
opt_fit = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
model_fit = Sequential()
model_fit.add(Dense(8, input_dim=4, activation='relu', kernel_initializer='ones', bias_initializer='ones'))
model_fit.add(Dense(3, activation='softmax', kernel_initializer='ones', bias_initializer='ones'))
model_fit.compile(loss='categorical_crossentropy', optimizer=opt_fit, metrics=keras.metrics.CategoricalAccuracy())

# Model for Tape:
##################################################################################################
opt_tape = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
model_tape = Sequential()
model_tape.add(Dense(8, input_dim=4, activation='relu', kernel_initializer='ones', bias_initializer='ones'))
model_tape.add(Dense(3, activation='softmax', kernel_initializer='ones', bias_initializer='ones'))
model_tape.compile(loss='categorical_crossentropy', optimizer=opt_tape, metrics=keras.metrics.CategoricalAccuracy())

EPOCHS = 200
SAMPLES = 150
BATCH_SIZE = 10

# Train with fit:
##################################################################################################
history = model_fit.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=False)

loss_fit = history.history['loss']
acc_fit = history.history['categorical_accuracy']

# print(loss_fit)
# print(acc_fit)

# Train with GradientTape:
##################################################################################################
loss_tape = []
acc_tape = []
batch_num = (int)(SAMPLES / BATCH_SIZE)

for i in range(EPOCHS):
    avg_loss_tape = 0
    avg_acc_tape = 0
    for j in range(batch_num):
        start_idx = BATCH_SIZE * j
        end_idx = BATCH_SIZE * (j + 1)
        X_batch = X[start_idx:end_idx]
        y_batch = Y[start_idx:end_idx]

        with tf.GradientTape() as tape:
            pred = model_tape(X_batch)
            loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, pred))
            grads = tape.gradient(loss, model_tape.trainable_weights)
            opt_tape.apply_gradients(zip(grads, model_tape.trainable_variables))
            avg_loss_tape += loss.numpy()
            avg_acc_tape += accuracy_score(np.argmax(y_batch, axis=1), np.argmax(pred, axis=1))

    avg_loss_tape = avg_loss_tape / batch_num
    avg_acc_tape = avg_acc_tape / batch_num
    print('Epoch:%d' % i, '\nloss: %.4f' % avg_loss_tape, " - categorical_accuracy: %.4f" % avg_acc_tape)
    loss_tape.append(avg_loss_tape)
    acc_tape.append(avg_acc_tape)

# Visualization:
##################################################################################################
epochs_range = range(EPOCHS)

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss_fit, label='Training with fit')
plt.plot(epochs_range, loss_tape, label='Training with Tape')
plt.legend(loc='upper right')
# plt.ylim(0, 50)
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc_fit, label='Training with fit')
plt.plot(epochs_range, acc_tape, label='Training with Tape')
plt.legend(loc='upper right')
# plt.ylim(0, 100)
plt.title('Training Accuracy')
plt.show()
