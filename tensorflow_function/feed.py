#coding=utf-8
import tensorflow as tf
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)


with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7.0], input2:[2.0]})


with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1:[7.0], input2:[2.0]})
    print type(result)
    print result

#the result output
#[array([ 14.], dtype=float32)]
#<type 'numpy.ndarray'>
#[ 14.]

