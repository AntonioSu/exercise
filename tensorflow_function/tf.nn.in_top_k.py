#coding=utf-8
#解释：这个函数的作用是返回一个布尔向量，说明目标值是否存在于预测值之中。
#输出数据是一个 targets 长度的布尔向量，如果目标值存在于预测值之中，那么 out[i] = true。
#注意：targets 是predictions中的索引位，并不是 predictions 中具体的值。
import tensorflow as tf  

predictions=tf.Variable(tf.truncated_normal([10,5],mean=0.0,stddev=1.0,dtype=tf.float32))  
targets=tf.Variable([1,1,1,1,1,1,1,1,1,1])  
eval_correct=tf.nn.in_top_k(predictions,targets,1)  
  
init=tf.initialize_all_variables()  
sess=tf.Session()  
sess.run(init)  
print sess.run(predictions)  
print sess.run(targets)  
print sess.run(eval_correct)  
sess.close()  

