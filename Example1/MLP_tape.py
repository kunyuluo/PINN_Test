from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras

housing_datasets = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(housing_datasets.data, housing_datasets.target)
x_train, x_validation, y_train, y_validation = train_test_split(x_train_full, y_train_full)

# print(x_train)
print(x_train.shape)
# print(y_train)
# print(y_train.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_validation = scaler.fit_transform(x_validation)
x_test = scaler.fit_transform(x_test)

model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)])

optimizer = keras.optimizers.SGD(lr=0.01)
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=optimizer)

samples = x_train.shape[0]
batch_size = 32
epochs = 20
# history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
#                     validation_data=(x_validation, y_validation))
batch_num = int(samples / batch_size)
print(batch_num)

for i in range(epochs):
    for j in range(batch_num):
        x_batch = x_train[j * batch_size: (j + 1) * batch_size]
        y_batch = y_train[j * batch_size: (j + 1) * batch_size]

        with tf.GradientTape() as tape:
            pred = model(x_batch)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(y_batch, pred))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f"Epoch {i}/{epochs}, Total Loss: {loss}")


mean_squared_error_test = model.evaluate(x_test, y_test)
print(mean_squared_error_test)

x_new = x_test[:3] # New instance
y_pred = model.predict(x_new)
print(y_pred)
