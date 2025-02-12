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
batch_num = int(len(y_train) / batch_size)
print('Number of batch: ', batch_num)


@tf.function
def train_step_v1(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.reduce_mean(objective(y, predictions))
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


@tf.function
def train_step_v2(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = objective(y, predictions)
        scaled_loss = optimizer_mp.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_weights)
    gradients = optimizer_mp.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


# Fit via gradient tape.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
loss_tape = []
for i in range(0, epochs):
    avg_loss = 0
    for step, (x_batch, y_batch) in enumerate(dataset):
        # print(x_batch.shape, y_batch.shape)
        loss = train_step_v2(x_batch, y_batch)
        avg_loss += loss.numpy()

    avg_loss = avg_loss / batch_num
    print(f"Epoch {i}/{epochs}, average loss: {avg_loss}")
    loss_tape.append(avg_loss)

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
