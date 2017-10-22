import os
import sys
import csv

sys.path.append("../")
from time import gmtime, strftime
import modules.features.data_representation as data_rep
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper
from tensorflow.python.training import moving_averages
import warnings
from timeit import default_timer as timer


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


if len(sys.argv) == 3:
    print('this method needs 3 args TRAINING_DIR CHECKPOINT_DIR REPORT_DIR')
    print('TRAINING_DIR ~> directory of target preprocessing files will be created by preprocessing.py')
    print('TESTING_DIR ~> directory of target preprocessing files will be created by preprocessing.py')
    print("MODEL_DIR ~> directory of model's will be stored")
else:

    # init
    training_dir = sys.argv[1]
    testing_dir = sys.argv[2]
    model_dir = sys.argv[3]
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    report_dir = os.path.join(model_dir, 'reports')
    log_dir = os.path.join(model_dir, 'logs')

    print(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    iteration = 300
    training_batch = 2
    testing_batch = 1
    num_cep = 264
    min_label_error_rate_diff = 0.005

    # property of Batch Normalization
    scale = 1
    offset = 0
    variance_epsilon = 0.001

    # property of weight
    mean = 1
    std = 0.046875
    relu_clip = 100
    n_hidden_1 = 128
    n_hidden_2 = 128
    n_hidden_3 = 2 * 128
    n_hidden_5 = 128
    n_hidden_6 = 25

    # property of BiRRN LSTM
    n_hidden_4 = 128
    forget_bias = 1

    # property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    decay = 0.999
    learning_rate = 0.001
    threshold = 0

    training_dataset = []
    target_training_dataset = []
    testing_dataset = []
    target_testing_dataset = []
    global_ler = []

    # load training dataset
    print("Loading training dataset")
    for root, dirs, files in os.walk(training_dir, topdown=False):
        for file in files:
            if file[0] != '_':
                print(os.path.join(training_dir, file))
                target_training_dataset.append(np.load(os.path.join(training_dir, '_' + file)))
                new_training_set = np.load(os.path.join(training_dir, file))
                training_dataset.append(new_training_set)

    # load testing dataset
    print("Loading testing dataset")
    for root, dirs, files in os.walk(testing_dir, topdown=False):
        for file in files:
            if file[0] != '_':
                print(os.path.join(testing_dir, file))
                target_testing_dataset.append(np.load(os.path.join(testing_dir, '_' + file)))
                new_testing_set = np.load(os.path.join(testing_dir, file))
                testing_dataset.append(new_testing_set)

    with tf.device('/cpu:0'):
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
            w1 = tf.Variable(tf.random_normal([num_cep, n_hidden_1], mean, std, tf.float32), name='fc1_w')
            b1 = tf.Variable(tf.random_normal([n_hidden_1], mean, std, tf.float32), name='fc1_b')

            h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(reshape_input_batch, w1), b1)), relu_clip)
            h1_bn = batch_norm(h1, 'fc1_bn', tf.cast(is_training, tf.bool))
            h1_dropout = tf.nn.dropout(h1_bn,1 - 0.05)


            w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean, std, tf.float32), name='fc2_w')
            b2 = tf.Variable(tf.random_normal([n_hidden_2], mean, std, tf.float32), name='fc2_b')

            h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1_dropout, w2), b2)), relu_clip)
            h2_bn = batch_norm(h2, 'fc2_bn', tf.cast(is_training, tf.bool))
            h2_dropout = tf.nn.dropout(h2_bn,1 - 0.05)


            w3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], mean, std, tf.float32), name='fc3_w')
            b3 = tf.Variable(tf.random_normal([n_hidden_3], mean, std, tf.float32), name='fc3_b')

            h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2_dropout, w3), b3)), relu_clip)
            h3_bn = batch_norm(h3, 'fc3_bn', tf.cast(is_training, tf.bool))
            h3_dropout = tf.nn.dropout(h3_bn,1 - 0.05)

        with tf.name_scope('biRNN'):
            # reshape to [batchsize x time x 2*n_hidden_4]
            # h3_dropout = tf.reshape(h3_dropout, [shape_input_batch[0], -1, n_hidden_3])

            # reshape to [time x batchsize x 2*n_hidden_4]
            h3_dropout = tf.reshape(h3_dropout, [-1, shape_input_batch[0], n_hidden_3])


            forward_cell_1 = BasicLSTMCell(n_hidden_4, forget_bias=forget_bias, state_is_tuple=True)
            forward_cell_1 = DropoutWrapper(forward_cell_1,1.0 - 0.0, 1.0 - 0.0)
            backward_cell_1 = BasicLSTMCell(n_hidden_4, forget_bias=forget_bias, state_is_tuple=True)
            backward_cell_1 = DropoutWrapper(backward_cell_1, 1.0 - 0.0, 1.0 - 0.0)
            # forward_cell_2 = BasicLSTMCell(n_hidden_5)
            # backward_cell_2 = BasicLSTMCell(n_hidden_5)

            # BiRNN
            # outputs, output_states_fw, output_states_bw = stack_bidirectional_dynamic_rnn(cells_fw=[forward_cell_1],
            #                                                                               cells_bw=[backward_cell_1],
            #                                                                               inputs=h3_dropout,
            #                                                                               dtype=tf.float32,
            #                                                                               sequence_length=seq_len,
            #                                                                               parallel_iterations=32,
            #                                                                               scope="biRNN")

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell_1,
                                                         cell_bw=backward_cell_1,
                                                         inputs=h3_dropout,
                                                         time_major=True,
                                                         sequence_length=seq_len,
                                                         dtype=tf.float32)

            outputs = tf.concat(outputs, 2)

            w5 = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_5], mean, std, tf.float32), name='fc5_w')
            b5 = tf.Variable(tf.random_normal([n_hidden_5], mean, std, tf.float32), name='fc5_b')
            # reshape to [batchsize * timestep x num_cepstrum]
            h4 = tf.reshape(outputs, [-1, 2 * n_hidden_4])
            h5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h4, w5), b5)), relu_clip)
            h5_bn = batch_norm(h5, 'fc5_bn', tf.cast(is_training, tf.bool))
            h5_dropout = tf.nn.dropout(h5_bn, 1.0 - 0.05)

        with tf.name_scope('logits'):
            w6 = tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6], mean, std, tf.float32), name='fc7_w')
            b6 = tf.Variable(tf.random_normal([n_hidden_6], mean, std, tf.float32), name='fc7_b')

            h6 = tf.add(tf.matmul(h5_dropout, w6), b6)

            # reshape to [time x batchsize x n_hidden_7]
            logits = tf.reshape(h6, [-1, shape_input_batch[0], n_hidden_6])

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

    #
    # RUN MODEL
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # Tensorboard
        writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
        last_iteration = 0
        writer.close()
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
            last_checkpoint = valid_dirs[last_iteration - 1]
            start = timer()
            print("Restoring " + os.path.join(os.path.join(checkpoint_dir, last_checkpoint)))
            # saving model state
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(os.path.join(checkpoint_dir, last_checkpoint), 'tensorflow_1.ckpt'))
            target_checkpoint_dir = os.path.join(os.path.join(checkpoint_dir, last_checkpoint))

            elapsed_time = timer() - start
            print("Elapsed time : " + str(elapsed_time))
        else:
            tf.global_variables_initializer().run()
            start = timer()
            print("Saving Base Model...")
            now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
            saver = tf.train.Saver()
            target_checkpoint_dir = os.path.join(checkpoint_dir, 'DMC-' + now)
            os.makedirs(target_checkpoint_dir)
            save_path = saver.save(sess, os.path.join(target_checkpoint_dir, 'tensorflow_1.ckpt'))
            elapsed_time = timer() - start
            print("Elapsed time : " + str(elapsed_time))
        _, num_cep = training_dataset[0].shape

        training_losses = []
        training_old_losses = []
        testing_losses = []
        testing_old_losses = []


        for iter in range(iteration):

            report_training = open(os.path.join(report_dir, 'report_training.txt'), "a")
            reporttrainingcsv = open(os.path.join(report_dir, 'result_training.csv'), "a")
            report_testing = open(os.path.join(report_dir, 'report_testing.txt'), "a")
            reporttestingcsv = open(os.path.join(report_dir, 'result_testing.csv'), "a")
            trainingcsvwriter = csv.writer(reporttrainingcsv)
            testingcsvwriter = csv.writer(reporttestingcsv)

            # =================================TRAINING PHASE=================================
            print("iteration #" + str(iter + last_iteration))
            report_training.write("iteration #" + str(iter + last_iteration) + '\n')
            csv_training_values = []
            csv_training_values.append(iter + last_iteration)
            if iter > 0:
                training_old_losses = training_losses
                training_losses = []

            training_shuffled_index = np.arange(len(training_dataset))
            np.random.shuffle(training_shuffled_index)

            for i in range(int(len(training_dataset) / int(training_batch))):
                start = timer()
                csv_training_values = []
                print("=================================TRAINING PHASE BATCH #" + str(i) + " ITERATION AT " + str(
                    iter) + "=================================")
                report_training.write(
                    "=================================TRAINING PHASE BATCH #" + str(i) + " ITERATION AT " + str(
                        iter) + "=================================" + '\n')
                csv_training_values.append(i)
                csv_training_values.append(learning_rate)
                # get batch shuffled index

                batch_i = []
                target = []
                for j in range(training_batch):
                    batch_i.append(training_dataset[training_shuffled_index[j + (i * training_batch)]])
                    target.append(target_training_dataset[training_shuffled_index[j + (i * training_batch)]])

                batch_i = data_rep.sparse_dataset(batch_i)
                print(batch_i.shape)
                # batch_i = training_dataset[(i*batch):(i*batch)+batch]
                sequence_length = np.array([batch_i.shape[1] for _ in range(training_batch)])

                # target = target_training_dataset[(i*batch):(i*batch)+batch]
                sparse_labels = data_rep.SimpleSparseTensorFrom(target)
                feed = {
                    input_batch: batch_i,
                    seq_len: sequence_length,
                    targets: sparse_labels,
                    is_training: True,
                    alpha: learning_rate
                }

                loss, logg, _ = sess.run([avg_loss, decode, optimizer], feed)
                print("Encoded CTC :")
                report_training.write("Encoded CTC :" + '\n')
                decode_text = data_rep.indices_to_text(logg[0][1])
                print(decode_text)
                print("first target : \n" + data_rep.indices_to_text(target[0]))
                report_training.write(decode_text + '\n')
                report_training.write("first target : " + data_rep.indices_to_text(target[0]) + '\n')
                csv_training_values.append(data_rep.indices_to_text(target[0]))
                #
                # summ = sess.run(summaries, feed)
                # writer.add_summary(summ,iter)

                print("negative log-probability :" + str(loss))
                report_training.write("negative log-probability :" + str(loss) + '\n')
                csv_training_values.append(loss)
                csv_training_values.append(decode_text)
                csv_training_values.append(data_rep.indices_to_text(target[0]))
                trainingcsvwriter.writerow(csv_training_values)
                training_losses.append(loss)

                elapsed_time = timer() - start
                cycle_batch = int(len(training_dataset) / int(training_batch))
                remaining_time = (((iteration - iter) * cycle_batch) - i) * elapsed_time
                print("Elapsed time : " + str(elapsed_time))
                report_training.write("Elapsed time: " + str(elapsed_time) + '\n')
                print("Remaining time : " + str(remaining_time))
                report_training.write("Remaining time: " + str(remaining_time) + '\n')

            if iter > 0:

                diff = np.array(training_losses) - np.array(training_old_losses)
                th = diff.mean()
                percentage = th / np.array(training_old_losses).mean() * 100
                print("Learning performance : " + str(th))
                report_training.write("Learning performance : " + str(th) + '\n')
                report_training.write("Learning percentage : " + str(percentage) + '\n')

                print("Saving ...")
                now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
                saver = tf.train.Saver()
                target_checkpoint_dir = os.path.join(checkpoint_dir, 'DMC-' + now)
                if not os.path.exists(target_checkpoint_dir):
                    os.makedirs(target_checkpoint_dir)
                save_path = saver.save(sess, os.path.join(target_checkpoint_dir, 'tensorflow_1.ckpt'))
                print("Checkpoint has been saved on path : " + str(save_path))
                report_training.write("Checkpoint has been saved on path : " + str(save_path) + '\n')

                _w1 = w1.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w1"), _w1)
                _b1 = b1.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b1"), _b1)
                _w2 = w2.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w2"), _w2)
                _b2 = b2.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b2"), _b2)
                _w3 = w3.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w3"), _w3)
                _b3 = b3.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b3"), _b3)
                _w6 = w6.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w6"), _w6)
                _b6 = b6.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b6"), _b6)
                _w7 = w7.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w7"), _w7)
                _b7 = b7.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b7"), _b7)



            else:
                print("Saving ...")
                now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
                saver = tf.train.Saver()
                target_checkpoint_dir = os.path.join(checkpoint_dir, 'DMC-' + now)
                os.makedirs(target_checkpoint_dir)
                save_path = saver.save(sess, os.path.join(target_checkpoint_dir, 'tensorflow_1.ckpt'))
                print("Checkpoint has been saved on path : " + str(save_path))
                report_training.write("Checkpoint has been saved on path : " + str(save_path) + '\n')

                _w1 = w1.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w1"), _w1)
                _b1 = b1.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b1"), _b1)
                _w2 = w2.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w2"), _w2)
                _b2 = b2.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b2"), _b2)
                _w3 = w3.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w3"), _w3)
                _b3 = b3.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b3"), _b3)
                _w6 = w6.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w6"), _w6)
                _b6 = b6.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b6"), _b6)
                _w7 = w7.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "w7"), _w7)
                _b7 = b7.eval(sess)
                np.savetxt(os.path.join(target_checkpoint_dir, "b7"), _b7)


            # =================================TESTING PHASE=================================

            print("iteration #" + str(iter + last_iteration))
            report_testing.write("iteration #" + str(iter + last_iteration) + '\n')
            print("moving avg")

            csv_testing_values = []
            csv_testing_values.append(iter + last_iteration)
            if iter > 0:
                testing_old_losses = testing_losses
                testing_losses = []

            testing_shuffled_index = np.arange(len(testing_dataset))
            np.random.shuffle(testing_shuffled_index)
            current_ler = []

            for i in range(int(len(testing_dataset) / int(testing_batch))):
                csv_testing_values = []
                print("=================================TESTING PHASE BATCH #" + str(
                    i) + "=================================")
                report_testing.write("=================================TESTING PHASE BATCH #" + str(
                    i) + "=================================" + '\n')

                csv_testing_values.append(i)
                csv_testing_values.append(learning_rate)
                # get batch shuffled index
                batch_i = []
                target = []
                for j in range(testing_batch):
                    batch_i.append(testing_dataset[testing_shuffled_index[j + (i * testing_batch)]])
                    target.append(target_testing_dataset[testing_shuffled_index[j + (i * testing_batch)]])

                batch_i = data_rep.sparse_dataset(batch_i)

                sequence_length = np.array([batch_i.shape[1] for _ in range(testing_batch)])

                sparse_labels = data_rep.SimpleSparseTensorFrom(target)
                feed = {
                    input_batch: batch_i,
                    seq_len: sequence_length,
                    targets: sparse_labels,
                    is_training: False,
                    alpha: 0
                }

                loss, logg, label_error_rate = sess.run([avg_loss, decode, ler], feed)
                current_ler.append(label_error_rate)
                print("Encoded CTC :")
                report_testing.write("Encoded CTC :" + '\n')
                decode_text = data_rep.indices_to_text(logg[0][1])
                print(decode_text)
                print("target : \n" + data_rep.indices_to_text(target[0]))
                report_testing.write(decode_text + '\n')
                report_testing.write("target : " + data_rep.indices_to_text(target[0]) + '\n')

                print("negative log-probability :" + str(loss))
                report_testing.write("negative log-probability :" + str(loss) + '\n')
                csv_testing_values.append(loss)
                csv_testing_values.append(decode_text)
                csv_testing_values.append(data_rep.indices_to_text(target[0]))
                testing_losses.append(loss)

                print("accuracy of label error rate")
                print("Label error rate : " + str(label_error_rate))
                report_testing.write("Label error rate : " + str(label_error_rate) + '\n')
                csv_testing_values.append(label_error_rate)
                testingcsvwriter.writerow(csv_testing_values)

            if (iter == 0):
                global_ler = np.array(current_ler)
                print("Current label error rate : " + str(global_ler.mean()))
                report_testing.write("Current label error rate : " + str(global_ler.mean()) + '\n')
            elif len(global_ler) > 0:
                current_ler = np.array(current_ler)
                avg_global_ler = global_ler.mean()
                avg_current_ler = current_ler.mean()
                print("Best label error rate : " + str(avg_global_ler))
                report_testing.write("Best label error rate : " + str(avg_global_ler) + '\n')
                print("Current label error rate : " + str(avg_current_ler))
                report_testing.write("Current label error rate : " + str(avg_current_ler) + '\n')
                if avg_global_ler - avg_current_ler < min_label_error_rate_diff:
                    warnings.warn(
                        "Training phase not learning, because the network doesn't seem to improved, consider to lowering the learning_rate (last label error rate is = " + str(
                            avg_current_ler) + ")")
                else:
                    global_ler = np.array(current_ler)

            pass



