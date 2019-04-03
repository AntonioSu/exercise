#coding=utf-8
import tensorflow as tf
a = tf.Variable(tf.random_normal([2,2],mean=0,stddev=1,seed=1))
b = tf.Variable(tf.truncated_normal([2,2],seed=2))
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
#在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
