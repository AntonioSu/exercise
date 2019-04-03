# -*- coding: utf-8 -*-

import tensorflow as tf;
import numpy as np;

c = np.random.random([10,1])
b = tf.nn.embedding_lookup(c, [1, 3])

with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print sess.run(b)
print c
#[[ 0.43675507]
#  [ 0.33368992]]
#[[ 0.8028126 ]
#  [ 0.43675507]
#  [ 0.77878843]
#  [ 0.33368992]
#  [ 0.38405982]
#  [ 0.89421374]
#  [ 0.08744736]
#  [ 0.19856621]
#  [ 0.19250068]
#  [ 0.92591149]]
