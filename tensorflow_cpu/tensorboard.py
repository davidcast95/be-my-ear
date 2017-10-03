
import tensorflow as tf

logs_path = '../logs'

#init
iteration = 5
batch = 8
num_cep = 286

# property of Batch Normalization
scale = 80
offset = 0
variance_epsilon = 0

#property of weight
mean = 0
std = 0.3
relu_clip = 80
n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 2 * 128

#init weight
with tf.device('/cpu:0'):
    with tf.name_scope('fc1'):
        w1 = tf.Variable(tf.random_normal([num_cep, n_hidden_1], mean, std, tf.float64), name='fc1_w')
        b1 = tf.Variable(tf.random_normal([n_hidden_1], mean, std, tf.float64), name='fc1_b')

    with tf.name_scope('fc2'):
        w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],mean,std,tf.float64),name='fc2_w')
        b2 = tf.Variable(tf.random_normal([n_hidden_2],mean,std,tf.float64),name='fc2_b')

    with tf.name_scope('fc3'):
        w3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],mean,std,tf.float64),name='fc3_w')
        b3 = tf.Variable(tf.random_normal([n_hidden_3],mean,std,tf.float64),name='fc3_b')


#SETUP NETWORK
input_training = tf.placeholder(tf.float64, [None, None, None], "input")

#batch normalization
# input_training = tf.nn.batch_normalization(input_training, mean, 0,offset,scale,variance_epsilon)

seq_len = tf.placeholder(tf.int32, [None], name="sequence_length")

# reshape to [batchsize * timestep x num_cepstrum]
training_batch = tf.reshape(input_training, [-1, num_cep])

#feed forward
h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(training_batch,w1), b1)),relu_clip)
h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1,w2), b2)),relu_clip)
h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2,w3), b3)),relu_clip)

writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
