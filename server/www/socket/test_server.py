import sys
sys.path.append("../../../")
import socket
import os
import numpy as np
import wave
from time import gmtime, strftime

#building model
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper
from timeit import default_timer as timer
import configparser as cp
import server.www.scripts.preprocessing as preproc

wav_data = '../wav_data'
preprocessed_data = '../preprocessed_data'
model_data = '../model'
type = 'mfcc'
num_context = 5


def batch_norm(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """

    return tf.cond(
        is_training,
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )


def batch_norm_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1:], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", 1, initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1:], initializer=tf.constant_initializer(0.0),
                                     trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1:], initializer=tf.constant_initializer(1.0),
                                     trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, list(range(len(shape) - 1)))
            avg = tf.cast(avg, tf.float32)
            var = tf.cast(var, tf.float32)
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output


config = cp.ConfigParser()
config.read('../config/model.ini')
# Reading from config
print("Read config model...")
# Properties of training
iteration = int(config['training']['iteration'])
training_batch = int(config['training']['training_batch'])
testing_batch = int(config['training']['testing_batch'])
print("Properties of training")
print("Iteration : " + str(iteration))
print("Training batch : " + str(training_batch))
print("Testing batch : " + str(testing_batch))
print("\n")

# Properties of weight
mean = float(config['init']['mean'])
std = float(config['init']['std'])
relu_clip = float(config['init']['relu_clip'])
print("Properties of Weight")
print("Mean : " + str(mean))
print("Std : " + str(std))
print("ReLU clip : " + str(relu_clip))
print('\n')

# Properties of Batch Normalization
print("Properties of Batch Normalization")
scale = float(config['batch-norm']['scale'])
offset = float(config['batch-norm']['offset'])
variance_epsilon = float(config['batch-norm']['variance_epsilon'])
decay = float(config['batch-norm']['decay'])
print("Scale : " + str(scale))
print("Offset : " + str(offset))
print("Variance epsilon : " + str(variance_epsilon))
print("Decay : " + str(decay))
print('\n')

# Properties of Forward Network
num_cep = int(config['forward-net']['num_cep'])
n_hidden_1 = int(config['forward-net']['n_hidden_1'])
n_hidden_2 = int(config['forward-net']['n_hidden_2'])
n_hidden_3 = int(config['forward-net']['n_hidden_3'])
print("Properties of Forward Network")
print("Num cepstrum : " + str(num_cep))
print("Hidden Layer 1 : " + str(n_hidden_1))
print("Hidden Layer 2 : " + str(n_hidden_2))
print("Hidden Layer 3 : " + str(n_hidden_3))
print('\n')

# Properties of Bidirectional RNN
n_hidden_4 = int(config['bi-rnn']['n_hidden_4'])
n_hidden_5 = int(config['bi-rnn']['n_hidden_5'])
forget_bias = int(config['bi-rnn']['forget_bias'])
print("Properties of Bidirectional RNN")
print("LSTM cell : " + str(n_hidden_4))
print("Forget bias : " + str(forget_bias))
print('\n')

# Properties of Classification Network
n_hidden_6 = int(config['classification-net']['n_hidden_6'])
n_hidden_7 = int(config['classification-net']['n_hidden_7'])
print("Properties of Classification Network")
print("Hidden Layer 5 : " + str(n_hidden_6))
print("Charset : " + str(n_hidden_7))
print('\n')

# property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
beta1 = float(config['adam']['beta1'])
beta2 = float(config['adam']['beta2'])
epsilon = float(config['adam']['epsilon'])
learning_rate = float(config['adam']['learning_rate'])
print('\n')

with tf.device('/gpu:0'):
    start = timer()
    print("Building the model")
    alpha = tf.Variable(0.001,name="alpha")
    is_training = tf.placeholder(tf.bool, name="is_training")

    # initialize input network
    input_batch = tf.placeholder(tf.float32, [None, None, None], "input")
    seq_len = tf.placeholder(tf.int32, [None], name="sequence_length")

    with tf.name_scope('forward-net'):
        shape_input_batch = tf.shape(input_batch)

        # Permute n_steps and batch_size
        transpose_input_batch = tf.transpose(input_batch, [1,0,2])

        # reshape to [batchsize * timestep x num_cepstrum]
        reshape_input_batch = tf.reshape(transpose_input_batch, [-1, num_cep])
        print(reshape_input_batch)
        w1 = tf.get_variable('fc1_w',[num_cep, n_hidden_1],tf.float32,tf.random_normal_initializer(mean,std))
        b1 = tf.get_variable('fc1_b',[n_hidden_1],tf.float32,tf.random_normal_initializer(mean,std))

        h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(reshape_input_batch, w1), b1)), relu_clip)
        h1_bn = batch_norm(h1, 'fc1_bn', tf.cast(is_training, tf.bool))
        h1_dropout = tf.nn.dropout(h1_bn,1 - 0.05)


        w2 = tf.get_variable('fc2_w',[n_hidden_1, n_hidden_2],tf.float32,tf.random_normal_initializer(mean,std))
        b2 = tf.get_variable('fc2_b',[n_hidden_2],tf.float32,tf.random_normal_initializer(mean,std))

        h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1_dropout, w2), b2)), relu_clip)
        h2_bn = batch_norm(h2, 'fc2_bn', tf.cast(is_training, tf.bool))
        h2_dropout = tf.nn.dropout(h2_bn,1 - 0.05)


        w3 = tf.get_variable('fc3_w',[n_hidden_2, n_hidden_3],tf.float32,tf.random_normal_initializer(mean,std))
        b3 = tf.get_variable('fc3_b',[n_hidden_3],tf.float32,tf.random_normal_initializer(mean,std))

        h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2_dropout, w3), b3)), relu_clip)
        h3_bn = batch_norm(h3, 'fc3_bn', tf.cast(is_training, tf.bool))
        h3_dropout = tf.nn.dropout(h3_bn,1 - 0.05)

    with tf.name_scope('biRNN'):
        # reshape to [batchsize x time x 2*n_hidden_4]
        # h3_dropout = tf.reshape(h3_dropout, [shape_input_batch[0], -1, n_hidden_3])

        # reshape to [time x batchsize x 2*n_hidden_4]
        h3_dropout = tf.reshape(h3_dropout, [-1, shape_input_batch[0], n_hidden_3])


        forward_cell_1 = BasicLSTMCell(n_hidden_4, forget_bias=1.0, state_is_tuple=True)
        forward_cell_1 = DropoutWrapper(forward_cell_1,1.0 - 0.0, 1.0 - 0.0)
        backward_cell_1 = BasicLSTMCell(n_hidden_4, forget_bias=1.0, state_is_tuple=True)
        backward_cell_1 = DropoutWrapper(backward_cell_1, 1.0 - 0.0, 1.0 - 0.0)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell_1,
                                                     cell_bw=backward_cell_1,
                                                     inputs=h3_dropout,
                                                     time_major=True,
                                                     sequence_length=seq_len,
                                                     dtype=tf.float32)

        outputs = tf.concat(outputs, 2)


        w6 = tf.get_variable('fc6_w',[n_hidden_3, n_hidden_6],tf.float32,tf.random_normal_initializer(mean,std))
        b6 = tf.get_variable('fc6_b',[n_hidden_6],tf.float32,tf.random_normal_initializer(mean,std))
        # reshape to [batchsize * timestep x num_cepstrum]
        h5 = tf.reshape(outputs, [-1, 2 * n_hidden_5])
        h6 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h5, w6), b6)), relu_clip)
        h6_bn = batch_norm(h6, 'fc6_bn', tf.cast(is_training, tf.bool))
        h6_dropout = tf.nn.dropout(h6_bn,1.0 - 0.05)

    with tf.name_scope('logits'):
        w7 = tf.get_variable('fc7_w',[n_hidden_6, n_hidden_7],tf.float32,tf.random_normal_initializer(mean,std))
        b7 = tf.get_variable('fc7_b',[n_hidden_7],tf.float32,tf.random_normal_initializer(mean,std))

        h7 = tf.add(tf.matmul(h6_dropout, w7), b7)
        # h7_bn = batch_norm(h7, 'fc7_bn', tf.cast(is_training, tf.bool))

        # reshape to [time x batchsize x n_hidden_7]
        logits = tf.reshape(h7, [-1, shape_input_batch[0], n_hidden_7])


    with tf.name_scope('decoder'):
        decode, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                         sequence_length=seq_len,
                                                         merge_repeated=True)

        targets = tf.sparse_placeholder(tf.int32, [None, None], name="target")

    with tf.name_scope('loss'):
        ctc_loss = tf.nn.ctc_loss(labels=targets,
                                  inputs=logits,
                                  sequence_length=seq_len)

        avg_loss = tf.reduce_mean(ctc_loss)
        tf.summary.histogram("avg_loss", avg_loss)

    with tf.name_scope('accuracy'):
        distance = tf.edit_distance(tf.cast(decode[0], tf.int32), targets)
        ler = tf.reduce_mean(distance, name='label_error_rate')

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)

        optimizer = optimizer.minimize(avg_loss)

    elapsed_time = timer() - start
    print("Elapsed time : " + str(elapsed_time))

# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
#     tf.global_variables_initializer().run()
#     # if os.path.exists(os.path.join(model_data, 'tensorflow_1.ckpt')):
#     start = timer()
#     print("Restoring " + os.path.join(os.path.join(model_data)))
#     # saving model state
#     saver = tf.train.Saver()
#     saver.restore(sess, os.path.join(model_data, 'tensorflow_1.ckpt'))
#
#     elapsed_time = timer() - start
#     print("Elapsed time : " + str(elapsed_time))


def run_model():
    datas = []
    for root, dirs, files in os.walk(preprocessed_data, topdown=False):
        for file in files:
            if (len(file.split('.')) == 2):
                filename, ext = file.split('.')
                if file[0] != '_' and ext == 'npy':
                    filename = os.path.join(root,file)
                    datas.append(np.load(filename))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        for data in datas:
            batch_i = preproc.data_rep.sparse_dataset([data])

            sequence_length = np.array([batch_i.shape[1] for _ in range(testing_batch)])
            target = np.array([1])

            sparse_labels = preproc.data_rep.SimpleSparseTensorFrom([target])
            feed = {
                input_batch: batch_i,
                seq_len: sequence_length,
                targets: sparse_labels,
                is_training: False,
                alpha: 0
            }
            print("run decode ")
            decoder = sess.run(decode, feed)
            decode_text = preproc.data_rep.indices_to_text(decoder[0][1])
            print(decode_text)


def preprocessing_data():
    for root, dirs, files in os.walk(wav_data, topdown=False):
        for file in files:
            print(file)
            if (len(file.split('.')) == 2):
                filename, ext = file.split('.')
                if (file[0] != '_' and ext == 'wav'):
                    if type == 'mfcc':
                        preproc.mffc_rep_wav(root,file,num_context)

    # run_model()

# create a socket object
serversocket = socket.socket(
    socket.AF_INET, socket.SOCK_STREAM)

# get local machine name
# host = socket.gethostbyname(socket.gethostname())
host = "192.168.1.5"
port = 14093

# bind to the port
serversocket.bind(('', port))
print('Listen from server ' + str(host) + ':' + str(port))


# listen to 1 request
serversocket.listen(5)

while True:

    # establish a connection
    clientsocket, addr = serversocket.accept()

    print("Got a connection from %s" % str(addr))

    now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    wavfile = wave.open(os.path.join('..','wav_data',now+'.wav'), 'w')
    wavfile.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
    length = 0

    while True:
        data = clientsocket.recv(1280)
        wavfile.writeframes(data)
        length += 1280
        print("duration : {}".format(str(length / 16000)))
        if (len(data) == 0):
            print("wav has been generated")
            wavfile.close()
            preprocessing_data()
            wavfile.close()
            print("saved : "+ str(now+'.wav'))
            now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
            wavfile = wave.open(os.path.join('..', 'wav_data', now + '.wav'), 'w')
            wavfile.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            length = 0
            break


