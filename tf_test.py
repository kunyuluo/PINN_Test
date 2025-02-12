import tensorflow as tf
import keras
import numpy as np

# test GradienTape
##########################################################################
# x1 = tf.Variable(3.0)
# x2 = tf.Variable(2.0)
#
# with tf.GradientTape() as tape:
#     y = x1**3 + 3*x1**2 + 3*x2**2
#
#     dy_dx = tape.gradient(y, x1)
# print(dy_dx)

# test tf.loss calculation
##########################################################################
# y_true = np.array([1, 2])
# # y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
# y_pred = np.array([[1], [2]])
# scce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# result = scce(y_true, y_pred)
# print(result)

# test tf.math
##########################################################################
# condition = np.array([True, False, True, False, True])
# y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# y_true = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
#
# # loss = tf.square(y_true - y_pred)
# # loss = tf.reduce_mean(loss)
# # print(loss)
#
# # result = tf.where(condition, y_pred, y_true)
# # result = tf.greater(y_pred, 5.0)
# result = tf.reduce_mean(y_pred)
# print(result)
a = tf.convert_to_tensor(3.0, dtype=tf.float32)
b = tf.convert_to_tensor(4.0, dtype=tf.float32)
const1 = tf.constant(2.0, dtype=tf.float32)
c = a*const1 + b
print(a)
print(b)
print(c)

# test GradienTape on function: y = 3*x1 + 5*x2
##########################################################################
# x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
# x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
# x1 = tf.Variable(x1)
# x2 = tf.Variable(x2)
# # print(x1)
#
# def model(x1, x2):
#     y = 3 * x1 + tf.math.pow(x1, 2) + 5 * x2
#     # y = 3 + x1
#     return y
#
#
# with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
#     tape.watch(x1)
#     y = model(x1, x2)
#     dy_dx1 = tape.gradient(y, x1)
#     # dy_dx1x1 = tape.gradient(dy_dx1, x1)
#     del tape
# # const = tf.constant(0.5, dtype='float64')
# # dy_dx1 = tf.subtract(dy_dx1, const)
# # dy_dx1x1 = tf.add(dy_dx1x1, const)
#
# print(dy_dx1)

# test tf.dataset
##########################################################################
# batch_size = 2
# x_train = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9],
#                  [6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13], [10, 11, 12, 13, 14]])
# y_train = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape(-1, 1)
# x1_train = x_train[:, 1]
#
# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
# x1_dataset = tf.data.Dataset.from_tensor_slices(x1_train).batch(batch_size)
#
# for step, (x_batch, y_batch) in enumerate(dataset):
#     print(step)
#     print(x_batch)
#     print(tf.reshape(x_batch[:, -1], [-1, 1]))

