#coding=utf_8

import tensorflow as tf
x=tf.constant([[1., 2.], [3., 4.]])
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print(sess.run(tf.reduce_mean(x)))
print(sess.run(tf.reduce_mean(x,0)))
print(sess.run(tf.reduce_mean(x,1)))

