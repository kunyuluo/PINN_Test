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

# Model for fit:
##################################################################################################
opt_fit = keras.optimizers.Adam()
loss_fit = keras.losses.MeanSquaredError()

model_fit = keras.Sequential([
    keras.layers.Dense(20, activation='selu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(10, activation='selu'),
    keras.layers.Dense(1)])
model_fit.compile(opt_fit, loss_fit)

# Model for Tape:
##################################################################################################
opt_tape = keras.optimizers.Adam()
loss_tape = keras.losses.MeanSquaredError()

model_tape = keras.Sequential([
    keras.layers.Dense(20, activation='selu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(10, activation='selu'),
    keras.layers.Dense(1)])
model_tape.compile(opt_tape, loss_tape)

batch_size = 4
epochs = 30
batch_num = int(len(y_train) / batch_size)

# Train with fit:
##################################################################################################
history = model_fit.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
losses_fit = history.history['loss']


# Train with GradientTape:
##################################################################################################
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model_tape(x, training=True)
        loss = tf.reduce_mean(loss_tape(y, predictions))
    gradients = tape.gradient(loss, model_tape.trainable_weights)
    opt_tape.apply_gradients(zip(gradients, model_tape.trainable_weights))
    return loss


# Fit via gradient tape.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
losses_tape = []
for i in range(0, epochs):
    avg_loss = 0
    for step, (x_batch, y_batch) in enumerate(dataset):
        loss = train_step(x_batch, y_batch)
        avg_loss += loss.numpy()

    avg_loss = avg_loss / batch_num
    print(f"Epoch {i}/{epochs}, average loss: {avg_loss}")
    losses_tape.append(avg_loss)

# Visualization:
##################################################################################################
epochs_range = range(epochs)

plt.figure(figsize=(14, 7))
plt.plot(epochs_range, losses_fit, label='Training with fit')
plt.plot(epochs_range, losses_tape, label='Training with Tape')
plt.legend(loc='upper right')
# plt.ylim(0, 50)
plt.title('Training Loss')
plt.show()

