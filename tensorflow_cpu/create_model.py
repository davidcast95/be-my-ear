import os
import sys
import csv
sys.path.append("../")
from time import gmtime, strftime
import modules.features.data_representation as data_rep
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell


if len(sys.argv) < 4:
    print ('this method needs 3 args WEIGHT_DIR TARGET_DIR DEPTH')
    print ('WEIGHT_DIR ~> directory of weight {w1,b1,w2,b2,w3,b3,w5,b5,w6,b6}')
    print ("TARGET_DIR ~> directory of base model will be stored")
    print ("DEPTH ~> depth of input example:129")
else:
    #init
    weight_dir = sys.argv[1]
    target_dir = sys.argv[2]
    iteration = 200
    batch = 1
    depths = int(sys.argv[3])

    mean = 0
    std = 0.3
    relu_clip = 20
    n_hidden_1 = 128
    n_hidden_2 = 128
    n_hidden_3 = 2 * 128
    n_hidden_4 = 128
    n_hidden_5 = 128
    n_hidden_6 = 28

    # property of BiRRN LSTM
    n_hidden_unit = 8 * 128
    forget_bias = 0

    # property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
    beta1 = 0.9
    beta2 = 0.9
    epsilon = 1e-6
    learning_rate = 0.001
    threshold = 0

    # init weight
    with tf.device('/cpu:0'):
        with tf.name_scope('fc1'):
            w1 = tf.Variable(tf.random_normal([depths, n_hidden_1], mean, std, tf.float64), name='fc1_w')
            b1 = tf.Variable(tf.random_normal([n_hidden_1], mean, std, tf.float64), name='fc1_b')

        with tf.name_scope('fc2'):
            w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean, std, tf.float64), name='fc2_w')
            b2 = tf.Variable(tf.random_normal([n_hidden_2], mean, std, tf.float64), name='fc2_b')

        with tf.name_scope('fc3'):
            w3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], mean, std, tf.float64), name='fc3_w')
            b3 = tf.Variable(tf.random_normal([n_hidden_3], mean, std, tf.float64), name='fc3_b')

        with tf.name_scope('fc5'):
            w5 = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_5], mean, std, tf.float64), name='fc5_w')
            b5 = tf.Variable(tf.random_normal([n_hidden_5], mean, std, tf.float64), name='fc5_b')

        with tf.name_scope('logits'):
            w6 = tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6], mean, std, tf.float64), name='logits_w')
            b6 = tf.Variable(tf.random_normal([n_hidden_6], mean, std, tf.float64), name='logits_b')

    # SETUP NETWORK
    input_training = tf.placeholder(tf.float64, [None, None, None], "input")

    seq_len = tf.placeholder(tf.int32, [None], name="sequence_length")

    # reshape to [batchsize * timestep x num_cepstrum]
    training_batch = tf.reshape(input_training, [-1, depths])

    # feed forward
    h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(training_batch, w1), b1)), relu_clip)
    h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1, w2), b2)), relu_clip)
    h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2, w3), b3)), relu_clip)

    # reshape to [time x batchsize x 2*n_hidden_4]
    h3 = tf.reshape(h3, [-1, batch, n_hidden_3])

    forward_cell = BasicLSTMCell(n_hidden_4, forget_bias, True)
    backward_cell = BasicLSTMCell(n_hidden_4, forget_bias, True)

    # BiRNN
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                             cell_bw=backward_cell,
                                                             inputs=h3,
                                                             dtype=tf.float64,
                                                             time_major=True,
                                                             sequence_length=seq_len)
    outputs = tf.concat(outputs, 2)

    # reshape to [batchsize * timestep x num_cepstrum]
    h4 = tf.reshape(outputs, [-1, 2 * n_hidden_4])

    # fully connected
    h5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h4, w5), b5)), relu_clip)

    h6 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h5, w6), b6)), relu_clip)

    # reshape to [time x batchsize x 2*n_hidden_4]
    logits = tf.reshape(h6, [-1, batch, n_hidden_6])
    logits = tf.cast(logits, tf.float32)

    decode, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                     sequence_length=seq_len,
                                                     merge_repeated=True)

    targets = tf.sparse_placeholder(tf.int32, [None, None], name="target")

    ctc_loss = tf.nn.ctc_loss(labels=targets,
                              inputs=logits,
                              sequence_length=seq_len)

    avg_loss = tf.reduce_mean(ctc_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    optimizer = optimizer.minimize(avg_loss)

    # RUN MODEL
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("Saving Base Model...")

        _w1 = np.load(os.path.join(weight_dir, "w1.npy"))
        if _w1.shape[0] != depths:
            _w1 = np.resize(_w1, (depths, n_hidden_1))
        w1.load(_w1, sess)
        _b1 = np.load(os.path.join(weight_dir, "b1.npy"))
        b1.load(_b1, sess)
        _w2 = np.load(os.path.join(weight_dir, "w2.npy"))
        w2.load(_w2, sess)
        _b2 = np.load(os.path.join(weight_dir, "b2.npy"))
        b2.load(_b2, sess)
        _w3 = np.load(os.path.join(weight_dir, "w3.npy"))
        w3.load(_w3, sess)
        _b3 = np.load(os.path.join(weight_dir, "b3.npy"))
        b3.load(_b3, sess)
        _w5 = np.load(os.path.join(weight_dir, "w5.npy"))
        w5.load(_w5, sess)
        _b5 = np.load(os.path.join(weight_dir, "b5.npy"))
        b5.load(_b5, sess)
        _w6 = np.load(os.path.join(weight_dir, "w6.npy"))
        w6.load(_w6, sess)
        _b6 = np.load(os.path.join(weight_dir, "b6.npy"))
        b6.load(_b6, sess)

        saver = tf.train.Saver()
        target_checkpoint_dir = os.path.join(target_dir, 'base-' + str(depths))
        os.makedirs(target_checkpoint_dir)
        save_path = saver.save(sess, os.path.join(target_checkpoint_dir, 'base.ckpt'))
