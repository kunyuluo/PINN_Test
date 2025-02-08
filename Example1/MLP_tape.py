from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

housing_datasets = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(housing_datasets.data, housing_datasets.target)
x_train, x_validation, y_train, y_validation = train_test_split(x_train_full, y_train_full)

# print(x_train)
print(x_train.shape)
# print(y_train)
print(y_train.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_validation = scaler.transform(x_validation)
x_test = scaler.transform(x_test)

model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)])

optimizer = keras.optimizers.SGD(lr=0.01)
optimizer_mp = keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=optimizer)

samples = x_train.shape[0]
batch_size = 32
epochs = 50
# history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
#                     validation_data=(x_validation, y_validation))
batch_num = int(samples / batch_size)
print('Number of batch: ', batch_num)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = tf.reduce_mean(keras.losses.mean_squared_error(y, pred))
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


@tf.function
def train_step_v2(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = tf.reduce_mean(keras.losses.mean_squared_error(y, pred))
        scaled_loss = optimizer_mp.get_scaled_loss(loss)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_weights)
    gradients = optimizer_mp.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size)

loss_tape = []
for i in range(epochs):
    avg_loss = 0
    # for j in range(batch_num):
    #     x_batch = x_train[j * batch_size: (j + 1) * batch_size]
    #     y_batch = y_train[j * batch_size: (j + 1) * batch_size]
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        # with tf.GradientTape() as tape:
        #     pred = model(x_batch)
        #     loss = tf.reduce_mean(keras.losses.mean_squared_error(y_batch, pred))
        #     gradients = tape.gradient(loss, model.trainable_variables)
        #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss = train_step_v2(x_batch, y_batch)
        avg_loss += loss.numpy()
        # print(avg_loss)

    avg_loss = avg_loss / batch_num
    print(f"Epoch {i}/{epochs}, average loss: {avg_loss}")
    loss_tape.append(avg_loss)

# mean_squared_error_test = model.evaluate(x_test, y_test)
# print(mean_squared_error_test)
#
# x_new = x_test[:3] # New instance
# y_pred = model.predict(x_new)
# print(y_pred)

# Visualization:
##################################################################################################
epochs_range = range(epochs)

plt.figure(figsize=(7, 7))
plt.plot(epochs_range, loss_tape, label='Training with Tape')
# plt.plot(epochs_range, loss_tape, label='Training with Tape')
plt.legend(loc='upper right')
# plt.ylim(0, 50)
plt.title('Training Loss')
plt.show()
