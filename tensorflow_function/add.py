import tensorflow as tf
hello = tf.constant('Hello,Tensorflow')
with tf.Session() as sess:
	print sess.run(hello)
	a=tf.constant(10)
	b=tf.constant(34)
	print sess.run(a+b)
