import tensorflow as tf

x1 = tf.Variable(3.0)
x2 = tf.Variable(2.0)

with tf.GradientTape() as tape:
    y = x1**3 + 3*x1**2 + 3*x2**2

    dy_dx = tape.gradient(y, x1)
print(dy_dx)
