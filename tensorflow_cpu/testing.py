import os
import sys
import csv
sys.path.append("../")
from time import gmtime, strftime
import modules.features.data_representation as data_rep
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.training import moving_averages

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
        moving_avg = tf.get_variable("moving_avg", shape[-1:], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1:], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, list(range(len(shape)-1)))
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

csv_fields = ['iteration','batch','learning_rate','ctc_loss','decoded_text']

if len(sys.argv) < 4:
    print ('this method needs 3 args TRAINING_DIR CHECKPOINT_DIR REPORT_DIR')
    print ('TESTING_DIR ~> directory of target preprocessing files will be created by preprocessing.py')
    print ("CHECKPOINT_DIR ~> directory of model's checkpoint will be stored")
    print ('REPORT_DIR ~> dicretory of result per checkpoint')
else:


    #init
    training_dir = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    report_dir = sys.argv[3]
    iteration = 1
    batch = 1
    num_cep = 286

    # property of Batch Normalization
    scale = 220
    offset = 0
    variance_epsilon = 0

    #property of weight
    mean = 0
    std = 0.3
    relu_clip = 220
    n_hidden_1 = 192
    n_hidden_2 = 192
    n_hidden_3 = 2 * 192
    n_hidden_5 = 192
    n_hidden_6 = 30

    #property of BiRRN LSTM
    n_hidden_4 = 192
    forget_bias = 0

    #property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.001
    decay = 0.999
    learning_rate = 0.001
    threshold = 0

    testing_dataset = []
    target_dataset = []

    # load training dataset
    for root, dirs, files in os.walk(training_dir, topdown=False):
        for file in files:
            if file[0] != '_':
                print(os.path.join(training_dir, file))
                target_dataset.append(np.load(os.path.join(training_dir, '_' + file)))
                new_testing_set = np.load(os.path.join(training_dir, file))
                testing_dataset.append(new_testing_set)

    testing_dataset = data_rep.sparse_dataset(testing_dataset)


    with tf.device('/cpu:0'):

        is_training = tf.placeholder(tf.bool, name="is_training")

        # initialize input network
        input_training = tf.placeholder(tf.float32, [None, None, None], "input")
        seq_len = tf.placeholder(tf.int32, [None], name="sequence_length")

        with tf.name_scope('forward-net'):
            # reshape to [batchsize * timestep x num_cepstrum]
            training_batch = tf.reshape(input_training, [-1, num_cep])
            with tf.variable_scope('fc1') as fc1:
                w1 = tf.Variable(tf.random_normal([num_cep, n_hidden_1], mean, std, tf.float32), name='fc1_w')
                b1 = tf.Variable(tf.random_normal([n_hidden_1], mean, std, tf.float32), name='fc1_b')

            h1 = tf.add(tf.matmul(training_batch,w1),b1)
            h1_bn = batch_norm(h1,'fc1_bn',tf.cast(is_training,tf.bool))
            h1 = tf.minimum(tf.nn.relu(h1_bn), relu_clip)

            with tf.variable_scope('fc2') as fc2:
                w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean, std, tf.float32), name='fc2_w')
                b2 = tf.Variable(tf.random_normal([n_hidden_2], mean, std, tf.float32), name='fc2_b')

            h2 = tf.add(tf.matmul(h1, w2), b2)
            h2_bn = batch_norm(h2, 'fc2_bn', tf.cast(is_training, tf.bool))
            h2 = tf.minimum(tf.nn.relu(h2_bn), relu_clip)

            with tf.variable_scope('fc3') as fc3:
                w3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], mean, std, tf.float32), name='fc3_w')
                b3 = tf.Variable(tf.random_normal([n_hidden_3], mean, std, tf.float32), name='fc3_b')

            h3 = tf.add(tf.matmul(h2, w3), b3)
            h3_bn = batch_norm(h3, 'fc3_bn', tf.cast(is_training, tf.bool))
            h3 = tf.minimum(tf.nn.relu(h3_bn), relu_clip)

        with tf.name_scope('biRNN'):
            with tf.variable_scope('fc4') as fc3:
                w5 = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_5], mean, std, tf.float32), name='fc5_w')
                b5 = tf.Variable(tf.random_normal([n_hidden_5], mean, std, tf.float32), name='fc5_b')
            # reshape to [time x batchsize x 2*n_hidden_4]
            h3 = tf.reshape(h3, [-1, batch, n_hidden_3])

            forward_cell =  GRUCell(n_hidden_4)
            backward_cell = GRUCell(n_hidden_4)

            # BiRNN
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                                cell_bw=backward_cell,
                                                                inputs=h3,
                                                                dtype=tf.float32,
                                                                time_major=True,
                                                                sequence_length=seq_len)
            outputs = tf.concat(outputs, 2)

            #reshape to [batchsize * timestep x num_cepstrum]
            h4 = tf.reshape(outputs,[-1,2 * n_hidden_4])
            h5 = tf.add(tf.matmul(h4, w5), b5)
            h5_bn = batch_norm(h5, 'fc5_bn', tf.cast(is_training, tf.bool))
            h5 = tf.minimum(tf.nn.relu(h5_bn), relu_clip)

        with tf.name_scope('logits'):
            with tf.variable_scope('fc6') as fc6:
                w6 = tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6], mean, std, tf.float32), name='fc6_w')
                b6 = tf.Variable(tf.random_normal([n_hidden_6], mean, std, tf.float32), name='fc6_b')

            h6 = tf.add(tf.matmul(h5, w6), b6)
            h6_bn = batch_norm(h6, 'fc6_bn', tf.cast(is_training, tf.bool))
            h6 = tf.minimum(tf.nn.relu(h6_bn), relu_clip)

            #reshape to [time x batchsize x n_hidden_6]
            logits = tf.reshape(h6, [-1, batch, n_hidden_6])
            logits = tf.cast(logits,tf.float32)

            tf.summary.histogram("logits",logits)

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

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=beta1,
                                               beta2=beta2,
                                               epsilon=epsilon)

            optimizer = optimizer.minimize(avg_loss)

        summaries = tf.summary.merge_all()


    # #Tensorboard
    # writer = tf.summary.FileWriter(report_dir,tf.get_default_graph())

    #
    #RUN MODEL
    with tf.Session() as sess:
        last_iteration = 0

        valid_dirs = []
        for root, dirs, files in os.walk(checkpoint_dir, topdown=False):
            for dir in dirs:
                if dir[0] == 'D' and dir[1] == 'M' and dir[2] == 'C' and dir[3] == '-':
                    valid_dirs.append(dir)
            if len(valid_dirs) > 0:
                break

        valid_dirs = sorted(valid_dirs)
        if len(valid_dirs) > 0:
            last_iteration = len(valid_dirs)
            last_checkpoint = valid_dirs[last_iteration-1]
            print ("Restoring " + os.path.join(os.path.join(checkpoint_dir,last_checkpoint)))
            # saving model state
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(os.path.join(checkpoint_dir,last_checkpoint),'tensorflow_1.ckpt'))
            target_checkpoint_dir = os.path.join(os.path.join(checkpoint_dir,last_checkpoint))

        else:
            tf.global_variables_initializer().run()
            print("Saving Base Model...")
            now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
            saver = tf.train.Saver()
            target_checkpoint_dir = os.path.join(checkpoint_dir, 'DMC-'+now)
            os.makedirs(target_checkpoint_dir)
            save_path = saver.save(sess, os.path.join(target_checkpoint_dir, 'tensorflow_1.ckpt'))

        _, num_cep = testing_dataset[0].shape

        losses = []
        old_losses = []
        report = open(os.path.join(report_dir,'report.txt'),"a")
        reportcsv = open(os.path.join(report_dir,'result.csv'),"a")
        losscsv = open(os.path.join(report_dir,'avg_loss.csv'),"a")
        csvwriter = csv.writer(reportcsv)
        csvloss = csv.writer(losscsv)
        for iter in range(iteration):
            if iter+last_iteration == 0:
                csvwriter.writerow(csv_fields)
            print ("iteration #"+str(iter + last_iteration))
            report.write("iteration #"+str(iter + last_iteration))
            csv_values = []
            csv_values.append(iter + last_iteration)
            if iter > 0:
                old_losses = losses
                losses = []
            for i in range(int(len(testing_dataset) / int(batch))):
                csv_values = []
                print ("batch #"+str(i))
                report.write("batch #"+str(i))
                csv_values.append(i)
                csv_values.append(learning_rate)
                #get batch
                batch_i = testing_dataset[(i*batch):(i*batch)+batch]
                sequence_length = np.array([batch_i.shape[1] for _ in range(batch)])
                target = target_dataset[(i*batch):(i*batch)+batch]
                sparse_labels = data_rep.SimpleSparseTensorFrom(target)
                feed = {
                    input_training : batch_i,
                    seq_len : sequence_length,
                    targets : sparse_labels,
                    is_training : False
                }

                logg = sess.run(decode, feed)
                print ("Encoded CTC :")
                report.write("Encoded CTC :")
                decode_text = data_rep.indices_to_text(logg[0][1])
                print(decode_text)
                report.write(decode_text)
                #
                # summ = sess.run(summaries, feed)
                # writer.add_summary(summ,iter)

                loss = sess.run(avg_loss, feed)
                print ("negative log-probability :" + str(loss))
                report.write("negative log-probability :" + str(loss))
                csvloss.writerow([loss])
                csv_values.append(loss)
                csv_values.append(decode_text)
                csvwriter.writerow(csv_values)
                losses.append(loss)

                sess.run(optimizer, feed)

            if iter > 0:

                diff = np.array(losses) - np.array(old_losses)
                th = diff.mean()
                percentage = th / np.array(old_losses).mean() * 100
                print ("Learning performance : " + str(th))
                report.write("Learning performance : " + str(th))
                report.write("Learning percentage : " + str(percentage))
