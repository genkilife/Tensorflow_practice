import matplotlib.image as mp_image
filename = "deep learning.jpg"
input_img = mp_image.imread(filename)

print("input dim = {}".format(input_img.ndim))
print("input shape = {}".format(input_img.shape))

import matplotlib.pyplot as plt
import tensorflow as tf
my_img = tf.placeholder("uint8", [None, None, 3])

slice_img = tf.slice(my_img, [100,0,0,], [160,-1,-1])

with tf.Session() as sess:
    result = sess.run(slice_img, feed_dict={my_img: input_img})
    print(result.shape)
    plt.imshow(result)
    plt.show()
