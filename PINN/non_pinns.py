import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate training data
np.random.seed(42)
num_samples = 1000
x_train = np.random.uniform(low=0, high=1, size=(num_samples, 1))
t_train = np.random.uniform(low=0, high=1, size=(num_samples, 1))
u_exact = np.sin(np.pi * x_train) * np.exp(-np.pi**2 * t_train)

# Convert to TensorFlow tensors
x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
t_train_tf = tf.convert_to_tensor(t_train, dtype=tf.float32)
u_exact_tf = tf.convert_to_tensor(u_exact, dtype=tf.float32)

# Combine x and t for training input
input_train = tf.concat([x_train_tf, t_train_tf], axis=1)

# Generate test data for prediction
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
t_test = np.linspace(0, 1, 100).reshape(-1, 1)

# Convert to TensorFlow tensors
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
t_test_tf = tf.convert_to_tensor(t_test, dtype=tf.float32)

input_test = tf.concat([x_test_tf, t_test_tf], axis=1)

u_exact_test = np.sin(np.pi * x_test) * np.exp(-np.pi**2 * t_test)
u_exact_test_tf = tf.convert_to_tensor(u_exact_test, dtype=tf.float32)

model = keras.Sequential([
    keras.layers.Dense(50, activation='tanh', input_dim=2),
    keras.layers.Dense(50, activation='tanh'),
    keras.layers.Dense(1)])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.mean_squared_error
model.compile(loss=loss, optimizer=optimizer)

# history = model.fit(input_train, u_exact_tf, epochs=500)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        # physics_loss_value = physics_loss(model, x_train_tf, t_train_tf)
        # data_loss_value = tf.reduce_mean(tf.square(model(input_train) - u_exact_tf))
        # total_loss = physics_loss_value + data_loss_value
        prediction = model(input_train, training=True)
        total_loss = keras.losses.mean_squared_error(u_exact_tf, prediction)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Total Loss: {total_loss.numpy()}")

y_pred = model.predict(input_test)

plt.plot(x_test, u_exact_test, label='Exact Solution')
plt.plot(x_test, y_pred, label='MLP Solution')
plt.legend()
plt.show()
