import tensorflow as tf  
import numpy as np  
  
A = [[1,3,4,5,6]]  
B = [[1,3,4,3,2]]  
  
with tf.Session() as sess:  
    print(sess.run(tf.equal(A, B)))  
