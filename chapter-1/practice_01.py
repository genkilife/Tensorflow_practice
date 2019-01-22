x = 1
y = x + 9
print(x)
print(y)

import tensorflow as tf
x = tf.constant(1, name='x')
y = tf.Variable(x+9, name='y')
print(x)
print(y)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
