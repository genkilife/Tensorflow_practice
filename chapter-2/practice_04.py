# Use gradient function

import tensorflow as tf

x = tf.placeholder(tf.float32)

y = 2*x*x

var_grad = tf.gradients(y,x)

with tf.Session() as sess:
    for i in range(10):
        var_grad_val = sess.run(var_grad, feed_dict={x:i})
        print(var_grad_val)
