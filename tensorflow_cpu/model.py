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
import warnings

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

if len(sys.argv) == 3:
    print ('this method needs 3 args TRAINING_DIR CHECKPOINT_DIR REPORT_DIR')
    print ('TRAINING_DIR ~> directory of target preprocessing files will be created by preprocessing.py')
    print ('TESTING_DIR ~> directory of target preprocessing files will be created by preprocessing.py')
    print ("MODEL_DIR ~> directory of model's will be stored")
else:


    #init
    training_dir = sys.argv[1]
    testing_dir = sys.argv[2]
    model_dir = sys.argv[3]
    checkpoint_dir = os.path.join(model_dir,'checkpoints')
    report_dir = os.path.join(model_dir,'reports')
    log_dir = os.path.join(model_dir,'logs')

    print(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    iteration = 100
    training_batch = 1
    testing_batch = 1
    num_cep = 247
    min_label_error_rate_diff = 0.005


    # property of Batch Normalization
    scale = 100
    offset = 0
    variance_epsilon = 0

    #property of weight
    mean = 0
    std = 0.3
    relu_clip = 100
    n_hidden_1 = 1024
    n_hidden_2 = 1024
    n_hidden_3 = 2 * 1024
    n_hidden_5 = 1024
    n_hidden_6 = 30

    #property of BiRRN LSTM
    n_hidden_4 = 1024
    forget_bias = 0

    #property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.001
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

    training_dataset = data_rep.sparse_dataset(training_dataset)
    testing_dataset = data_rep.sparse_dataset(testing_dataset)



    with tf.device('/cpu:0'):

        is_training = tf.placeholder(tf.bool, name="is_training")

        # initialize input network
        input_batch = tf.placeholder(tf.float32, [None, None, None], "input")
        seq_len = tf.placeholder(tf.int32, [None], name="sequence_length")

        with tf.name_scope('forward-net'):
            # reshape to [batchsize * timestep x num_cepstrum]
            reshape_input_batch = tf.reshape(input_batch, [-1, num_cep])
            with tf.variable_scope('fc1') as fc1:
                w1 = tf.Variable(tf.random_normal([num_cep, n_hidden_1], mean, std, tf.float32), name='fc1_w')
                b1 = tf.Variable(tf.random_normal([n_hidden_1], mean, std, tf.float32), name='fc1_b')

            h1 = tf.add(tf.matmul(reshape_input_batch,w1),b1)
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
            h3 = tf.cond(is_training,
                         lambda: tf.reshape(h3, [-1, training_batch, n_hidden_3]),
                         lambda: tf.reshape(h3, [-1, testing_batch, n_hidden_3]) )

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
            logits = tf.cond(is_training,
                             lambda :tf.reshape(h6, [-1, training_batch, n_hidden_6]),
                             lambda :tf.reshape(h6, [-1, testing_batch, n_hidden_6]))
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

        with tf.name_scope('accuracy'):
        	distance = tf.edit_distance(tf.cast(decode[0], tf.int32), targets)
        	ler = tf.reduce_mean(distance, name='label_error_rate')

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=beta1,
                                               beta2=beta2,
                                               epsilon=epsilon)

            optimizer = optimizer.minimize(avg_loss)



        summaries = tf.summary.merge_all()



    #
    #RUN MODEL
    with tf.Session() as sess:
        #Tensorboard
        writer = tf.summary.FileWriter(log_dir,graph=sess.graph)
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

            #=================================TRAINING PHASE=================================
            if iter+last_iteration == 0:
                trainingcsvwriter.writerow(csv_fields)
            print ("iteration #"+str(iter + last_iteration))
            report_training.write("iteration #" + str(iter + last_iteration) + '\n')
            csv_training_values = []
            csv_training_values.append(iter + last_iteration)

            if iter > 0:
                training_old_losses = training_losses
                training_losses = []

            shuffled_index = np.arange(len(training_dataset))
            np.random.shuffle(shuffled_index)

            for i in range(int(len(training_dataset) / int(training_batch))):
                csv_training_values = []
                print ("=================================TRAINING PHASE BATCH #"+str(i)+" ITERATION AT "+str(iter)+"=================================")
                report_training.write("=================================TRAINING PHASE BATCH #"+str(i)+" ITERATION AT "+str(iter)+"=================================" + '\n')
                csv_training_values.append(i)
                csv_training_values.append(learning_rate)
                #get batch shuffled index
                batch_i = []
                target = []
                for j in range(training_batch):
                    batch_i.append(training_dataset[shuffled_index[j + (i * training_batch)]])
                    target.append(target_training_dataset[shuffled_index[j + (i * training_batch)]])
                batch_i = np.array(batch_i)
                target = np.array(target)
                # batch_i = training_dataset[(i*batch):(i*batch)+batch]
                sequence_length = np.array([batch_i.shape[1] for _ in range(training_batch)])

                # target = target_training_dataset[(i*batch):(i*batch)+batch]
                sparse_labels = data_rep.SimpleSparseTensorFrom(target)
                feed = {
                    input_batch : batch_i,
                    seq_len : sequence_length,
                    targets : sparse_labels,
                    is_training : True
                }

                loss, logg, _ = sess.run([avg_loss, decode, optimizer], feed)
                print ("Encoded CTC :")
                report_training.write("Encoded CTC :" + '\n')
                decode_text = data_rep.indices_to_text(logg[0][1])
                print(decode_text)
                print("target : \n" + data_rep.indices_to_text(target[0]))
                report_training.write(decode_text + '\n')
                report_training.write("target : " + data_rep.indices_to_text(target[0]) + '\n')
                csv_training_values.append(target)
                #
                # summ = sess.run(summaries, feed)
                # writer.add_summary(summ,iter)

                print ("negative log-probability :" + str(loss))
                report_training.write("negative log-probability :" + str(loss) + '\n')
                csv_training_values.append(loss)
                csv_training_values.append(decode_text)
                csv_training_values.append(data_rep.indices_to_text(target[0]))
                trainingcsvwriter.writerow(csv_training_values)
                training_losses.append(loss)


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
                os.makedirs(target_checkpoint_dir)
                save_path = saver.save(sess, os.path.join(target_checkpoint_dir, 'tensorflow_1.ckpt'))
                print("Checkpoint has been saved on path : " + str(save_path))
                report_training.write("Checkpoint has been saved on path : " + str(save_path) + '\n')

                _w1 = w1.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w1"), _w1)
                _b1 = b1.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b1"), _b1)
                _w2 = w2.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w2"), _w2)
                _b2 = b2.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b2"), _b2)
                _w3 = w3.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w3"), _w3)
                _b3 = b3.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b3"), _b3)
                _fw_w = sess.run(forward_cell.variables)[0]
                np.save(os.path.join(target_checkpoint_dir, "fw_w"), _fw_w)
                _bw_w = sess.run(backward_cell.variables)[0]
                np.save(os.path.join(target_checkpoint_dir, "bw_w"), _bw_w)
                _w5 = w5.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w5"), _w5)
                _b5 = b5.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b5"), _b5)
                _w6 = w6.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w6"), _w6)
                _b6 = b6.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b6"), _b6)



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
                np.save(os.path.join(target_checkpoint_dir, "w1"), _w1)
                _b1 = b1.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b1"), _b1)
                _w2 = w2.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w2"), _w2)
                _b2 = b2.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b2"), _b2)
                _w3 = w3.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w3"), _w3)
                _b3 = b3.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b3"), _b3)
                _fw_w = sess.run(forward_cell.variables)[0]
                np.save(os.path.join(target_checkpoint_dir, "fw_w"), _fw_w)
                _bw_w = sess.run(backward_cell.variables)[0]
                np.save(os.path.join(target_checkpoint_dir, "bw_w"), _bw_w)
                _w5 = w5.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w5"), _w5)
                _b5 = b5.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b5"), _b5)
                _w6 = w6.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "w6"), _w6)
                _b6 = b6.eval(sess)
                np.save(os.path.join(target_checkpoint_dir, "b6"), _b6)

            # =================================TESTING PHASE=================================

            if iter + last_iteration == 0:
                testingcsvwriter.writerow(csv_fields)
            print("iteration #" + str(iter + last_iteration))
            report_testing.write("iteration #" + str(iter + last_iteration) + '\n')
            csv_testing_values = []
            csv_testing_values.append(iter + last_iteration)
            if iter > 0:
                testing_old_losses = testing_losses
                testing_losses = []

            shuffled_index = np.arange(len(testing_dataset))
            np.random.shuffle(shuffled_index)
            current_ler = []

            for i in range(int(len(testing_dataset) / int(testing_batch))):
                csv_testing_values = []
                print ("=================================TESTING PHASE BATCH #"+str(i)+"=================================")
                report_testing.write("=================================TESTING PHASE BATCH #"+str(i)+"=================================" + '\n')

                csv_testing_values.append(i)
                csv_testing_values.append(learning_rate)
                # get batch shuffled index
                batch_i = []
                target = []
                for j in range(testing_batch):
                    batch_i.append(testing_dataset[shuffled_index[j + (i * testing_batch)]])
                    target.append(target_testing_dataset[shuffled_index[j + (i * testing_batch)]])
                batch_i = np.array(batch_i)
                target = np.array(target)

                sequence_length = np.array([batch_i.shape[1] for _ in range(testing_batch)])

                sparse_labels = data_rep.SimpleSparseTensorFrom(target)
                feed = {
                    input_batch: batch_i,
                    seq_len: sequence_length,
                    targets: sparse_labels,
                    is_training: False
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

            if len(global_ler) > 0:
                current_ler = np.array(current_ler)
                avg_global_ler = global_ler.mean()
                avg_current_ler = current_ler.mean()
                if avg_global_ler - avg_current_ler < min_label_error_rate_diff:
                    warnings.warn("Training phase stop, because the network doesn't seem to improved, consider to lowering the learning_rate (last label error rate is = " + str(avg_current_ler) + ")")

            global_ler = np.array(current_ler)
            
            pass



