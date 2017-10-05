
import tensorflow as tf

logs_path = '../logs'

inputs = tf.Variable(tf.random_normal([1,5],0,1,tf.float32),name="input")

#init weight
with tf.device('/cpu:0'):
    with tf.name_scope('fc1'):
        w1 = tf.Variable(tf.random_normal([5, 5], 0, 1, tf.float32), name='fc1_w')
        b1 = tf.Variable(tf.random_normal([5], 0, 1, tf.float32), name='fc1_b')
#feed forward
h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(inputs,w1), b1)),20)
tf.summary.histogram("h1",h1)

tf.summary.scalar("inputs",5)
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
