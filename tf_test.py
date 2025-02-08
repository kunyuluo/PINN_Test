import tensorflow as tf
import keras
import numpy as np

# x1 = tf.Variable(3.0)
# x2 = tf.Variable(2.0)
#
# with tf.GradientTape() as tape:
#     y = x1**3 + 3*x1**2 + 3*x2**2
#
#     dy_dx = tape.gradient(y, x1)
# print(dy_dx)

y_true = np.array([1, 2])
# y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
y_pred = np.array([[1], [2]])
scce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
result = scce(y_true, y_pred)
print(result)
