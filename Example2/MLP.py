from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

tf.random.set_seed(0)

x_train, y_train = load_diabetes(return_X_y=True)
# print(x_train.shape, y_train.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

model = keras.Sequential([
    keras.layers.Dense(20, activation='selu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(10, activation='selu'),
    keras.layers.Dense(1)])

# Training parameters.
optimizer = keras.optimizers.Adam()
optimizer_mp = keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
objective = keras.losses.MeanSquaredError()

batch_size = 4
epochs = 30

# Fit via fit method.
model.compile(optimizer, objective)
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
loss_fit = history.history['loss']

# Visualization:
##################################################################################################
epochs_range = range(epochs)

plt.figure(figsize=(7, 7))
plt.plot(epochs_range, loss_fit, label='Training with fit')
# plt.plot(epochs_range, loss_tape, label='Training with Tape')
plt.legend(loc='upper right')
# plt.ylim(0, 50)
plt.title('Training Loss')
plt.show()
