import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

df = pd.read_csv('Dummy_data.csv')
df = df.sample(frac=1)

y = df['y'].values.reshape(-1, 1)
X = df.drop('y', axis='columns').to_numpy()
x1 = df['x1'].values.reshape(-1, 1)
x2 = df['x2'].values.reshape(-1, 1)
x3 = df['x3'].values.reshape(-1, 1)
x4 = df['x4'].values.reshape(-1, 1)
x5 = df['x5'].values.reshape(-1, 1)

x_train = np.concatenate([x1, x2, x3, x4, x5], axis=1)
y_train = y

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
x1_train_tf = tf.reshape(x_train_tf[:, 0], [-1, 1])
x2_train_tf = tf.reshape(x_train_tf[:, 1], [-1, 1])
x3_train_tf = tf.reshape(x_train_tf[:, 2], [-1, 1])
x4_train_tf = tf.reshape(x_train_tf[:, 3], [-1, 1])
x5_train_tf = tf.reshape(x_train_tf[:, 4], [-1, 1])

# x_batch = x_train_tf[:16]
# x2_batch = x2_train_tf[:16]
# print(x_batch)
# print(x2_batch)

# Define the model:
######################################################################################
my_model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)])

optimizer = keras.optimizers.SGD(lr=0.01)
optimizer_mp = keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
# model.compile(loss=keras.losses.mean_squared_error, optimizer=optimizer)


# Define the loss function (physics-informed loss)
def physics_loss(model, x1, x2, x3, x4, x5):
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(x2)
        y_pred = model(tf.concat([x1, x2, x3, x4, x5], axis=1))
        y_x2 = tape.gradient(y_pred, x2)
        del tape

    alpha = tf.constant(0.3, dtype='float32')

    # Define the heat equation PDE
    pde_loss = y_x2 - alpha

    return tf.reduce_mean(tf.square(pde_loss))


@tf.function
def train_step(model, batch_index, x1, x2, x3, x4, x5, x, y):
    start_idx = batch_size * batch_index
    end_idx = batch_size * (batch_index + 1)
    x_batch = x[start_idx:end_idx]
    y_batch = y[start_idx:end_idx]
    x1_batch = x1[start_idx:end_idx]
    x2_batch = x2[start_idx:end_idx]
    x3_batch = x3[start_idx:end_idx]
    x4_batch = x4[start_idx:end_idx]
    x5_batch = x5[start_idx:end_idx]

    with tf.GradientTape() as tape1:
        # y_pred = my_model(x_batch)

        # Physics Loss:
        # physics_loss_value = physics_loss(my_model, x1_batch, x2_batch, x3_batch, x4_batch, x5_batch)
        with tf.GradientTape(persistent=False) as tape2:
            tape2.watch(x2_batch)
            y_pred = model(tf.concat([x1_batch, x2_batch, x3_batch, x4_batch, x5_batch], axis=1))
            y_x2 = tape2.gradient(y_pred, x2_batch)
            del tape2

        alpha = tf.constant(0.3, dtype='float32')

        # Define the heat equation PDE
        pde_loss = y_x2 - alpha
        physics_loss_value = tf.reduce_mean(tf.square(pde_loss))

        # Data Loss:
        data_pred = model(x_batch)
        data_loss_value = tf.reduce_mean(tf.square(y_batch - data_pred))

        # Total Loss:
        total_loss_value = physics_loss_value * pde_coeff + data_loss_value * data_coeff

    gradients = tape1.gradient(total_loss_value, my_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, my_model.trainable_weights))

    return physics_loss_value, data_loss_value, total_loss_value


# Training Parameters:
######################################################################################
samples = x_train.shape[0]
batch_size = 16
epochs = 50

batch_num = int(samples / batch_size)
print('Number of batch: ', batch_num)

# Train with GradientTape:
######################################################################################
data_loss_tape = []
physics_loss_tape = []
total_loss_tape = []
pde_coeff = tf.constant(1e1, dtype='float32')
data_coeff = tf.constant(1e1, dtype='float32')
for i in range(epochs):
    avg_data_loss = 0
    avg_physics_loss = 0
    avg_total_loss = 0
    for j in range(batch_num):
        # start_idx = batch_size * j
        # end_idx = batch_size * (j + 1)
        # x_batch = x_train_tf[start_idx:end_idx]
        # y_batch = y_train_tf[start_idx:end_idx]
        # x1_batch = x1_train_tf[start_idx:end_idx]
        # x2_batch = x2_train_tf[start_idx:end_idx]
        # x3_batch = x3_train_tf[start_idx:end_idx]
        # x4_batch = x4_train_tf[start_idx:end_idx]
        # x5_batch = x5_train_tf[start_idx:end_idx]
        #
        # with tf.GradientTape() as tape:
        #     y_pred = my_model(x_batch)
        #     physics_loss_value = physics_loss(my_model, x1_batch, x2_batch, x3_batch, x4_batch, x5_batch)
        #     data_loss_value = tf.reduce_mean(tf.square(y_batch - y_pred))
        #     total_loss_value = physics_loss_value * pde_coeff + data_loss_value * data_coeff
        #     # total_loss_value = physics_loss_value + data_loss_value
        #
        # gradients = tape.gradient(total_loss_value, my_model.trainable_weights)
        # optimizer.apply_gradients(zip(gradients, my_model.trainable_weights))
        physics_loss_value, data_loss_value, total_loss_value = train_step(my_model, j, x1_train_tf, x2_train_tf,
                                                                           x3_train_tf, x4_train_tf, x5_train_tf,
                                                                           x_train_tf, y_train_tf)

        avg_data_loss += data_loss_value.numpy()
        avg_physics_loss += physics_loss_value.numpy()
        avg_total_loss += total_loss_value.numpy()

    avg_data_loss = avg_data_loss / batch_num
    avg_physics_loss = avg_physics_loss / batch_num
    avg_total_loss = avg_total_loss / batch_num

    data_loss_tape.append(avg_data_loss)
    physics_loss_tape.append(avg_physics_loss)
    total_loss_tape.append(avg_total_loss)

    print(f"Epoch {i}/{epochs}, Total Loss: {round(avg_total_loss, 3)}, "
          f"Physics Loss: {round(avg_physics_loss, 3)}, Data Loss: {round(avg_data_loss, 3)}")

# Visualization:
##################################################################################################
# epochs_range = range(epochs)
#
# plt.figure(figsize=(7, 7))
# plt.plot(epochs_range, data_loss_tape, label='Data Loss')
# plt.plot(epochs_range, physics_loss_tape, label='Physics Loss')
# plt.plot(epochs_range, total_loss_tape, label='Total Loss')
# plt.legend(loc='upper right')
# # plt.ylim(0, 50)
# plt.title('Training Loss')
# plt.show()

