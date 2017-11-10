import os
import sys
import csv

sys.path.append("../")
import numpy as np
import configparser as cp
from modules.model.session import session
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper
from tensorflow.python.training import moving_averages
import modules.features.data_representation as data_rep

class Model:
    def __init__(self):
        self.device = "/cpu:0"
        self.training_batch = 2
        self.testing_batch = 1
        self.num_cep = 264

        # property of Batch Normalization
        self.scale = 1
        self.offset = 0
        self.variance_epsilon = 0.001
        self.decay = 0.999

        # property of weight
        self.mean = 1
        self.std = 0.046875
        self.relu_clip = 100
        self.n_hidden_1 = 128
        self.n_hidden_2 = 128
        self.n_hidden_3 = 2 * 128
        self.n_hidden_5 = 128
        self.n_hidden_6 = 25

        # property of BiRRN LSTM
        self.n_hidden_4 = 128
        self.forget_bias = 1

        # property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate = 0.001


    def load_from_file(self,file_config):
        session.start_timer()
        print("Reading to " + file_config)
        # Reading from config
        config = cp.ConfigParser()
        config.read(file_config)

        # Properties of training
        self.iteration = int(config['training']['iteration'])
        self.training_batch = int(config['training']['training_batch'])
        self.testing_batch = int(config['training']['testing_batch'])

        # Properties of weight
        self.mean = float(config['init']['mean'])
        self.std = float(config['init']['std'])
        self.relu_clip = float(config['init']['relu_clip'])


        # Properties of Batch Normalization
        self.scale = float(config['batch-norm']['scale'])
        self.offset = float(config['batch-norm']['offset'])
        self.variance_epsilon = float(config['batch-norm']['variance_epsilon'])
        self.decay = float(config['batch-norm']['decay'])

        # Properties of Forward Network
        self.num_cep = int(config['forward-net']['num_cep'])
        self.n_hidden_1 = int(config['forward-net']['n_hidden_1'])
        self.n_hidden_2 = int(config['forward-net']['n_hidden_2'])
        self.n_hidden_3 = int(config['forward-net']['n_hidden_3'])
        self.n_hidden_5 = int(config['forward-net']['n_hidden_5'])

        # Properties of Bidirectional RNN
        self.n_hidden_4 = int(config['bi-rnn']['n_hidden_4'])
        self.forget_bias = int(config['bi-rnn']['forget_bias'])

        # Properties of Classification Network
        self.n_hidden_6 = int(config['classification-net']['n_hidden_6'])

        # property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
        self.beta1 = float(config['adam']['beta1'])
        self.beta2 = float(config['adam']['beta2'])
        self.epsilon = float(config['adam']['epsilon'])
        self.learning_rate = float(config['adam']['learning_rate'])

        session.stop_timer()
        print("Read config model...")

        print("Properties of training")
        print("Iteration : " + str(self.iteration))
        print("Training batch : " + str(self.training_batch))
        print("Testing batch : " + str(self.testing_batch))
        print("\n")

        print("Properties of Weight")
        print("Mean : " + str(self.mean))
        print("Std : " + str(self.std))
        print("ReLU clip : " + str(self.relu_clip))
        print('\n')

        print("Properties of Batch Normalization")
        print("Scale : " + str(self.scale))
        print("Offset : " + str(self.offset))
        print("Variance epsilon : " + str(self.variance_epsilon))
        print("Decay : " + str(self.decay))
        print('\n')

        print("Properties of Forward Network")
        print("Num cepstrum : " + str(self.num_cep))
        print("Hidden Layer 1 : " + str(self.n_hidden_1))
        print("Hidden Layer 2 : " + str(self.n_hidden_2))
        print("Hidden Layer 3 : " + str(self.n_hidden_3))
        print('\n')


        print("Properties of Bidirectional RNN")
        print("LSTM cell : " + str(self.n_hidden_4))
        print("Forget bias : " + str(self.forget_bias))
        print('\n')

        print("Properties of Classification Network")
        print("Hidden Layer 5 : " + str(self.n_hidden_5))
        print("Charset : " + str(self.n_hidden_6))
        print('\n')

        print("Properties of Adam Optimizer")
        print("Beta 1 : " + str(self.beta1))
        print("Beta 2 : " + str(self.beta2))
        print("Epsilon : " + str(self.epsilon))
        print("Learning rate : " + str(self.learning_rate))
        print('\n')


    def batch_norm(self, x, scope, is_training, epsilon=0.001, decay=0.99):
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
            lambda: self.batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
            lambda: self.batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
        )


    def batch_norm_layer(self, x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
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

    def create_network(self):
        with tf.device(self.device):
            session.start_timer()
            print("Building the model")
            self.alpha = tf.Variable(0.001, name="alpha")
            self.is_training = tf.placeholder(tf.bool, name="is_training")

            # initialize input network
            self.input_batch = tf.placeholder(tf.float32, [None, None, None], "input")
            self.seq_len = tf.placeholder(tf.int32, [None], name="sequence_length")

            with tf.name_scope('forward-net'):
                self.shape_input_batch = tf.shape(self.input_batch)

                # Permute n_steps and batch_size
                self.transpose_input_batch = tf.transpose(self.input_batch, [1, 0, 2])

                # reshape to [batchsize * timestep x num_cepstrum]
                self.reshape_input_batch = tf.reshape(self.transpose_input_batch, [-1, self.num_cep])
                self.w1 = tf.get_variable('fc1_w', [self.num_cep, self.n_hidden_1], tf.float32, tf.random_normal_initializer(self.mean, self.std))
                self.b1 = tf.get_variable('fc1_b', [self.n_hidden_1], tf.float32, tf.random_normal_initializer(self.mean, self.std))

                self.h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(self.reshape_input_batch, self.w1), self.b1)), self.relu_clip)
                self.h1_bn = self.batch_norm(self.h1, 'fc1_bn', tf.cast(self.is_training, tf.bool))
                # self.h1_dropout = tf.nn.dropout(self.h1_bn,1 - 0.05)


                self.w2 = tf.get_variable('fc2_w', [self.n_hidden_1, self.n_hidden_2], tf.float32, tf.random_normal_initializer(self.mean, self.std))
                self.b2 = tf.get_variable('fc2_b', [self.n_hidden_2], tf.float32, tf.random_normal_initializer(self.mean, self.std))

                self.h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(self.h1_bn, self.w2), self.b2)), self.relu_clip)
                self.h2_bn = self.batch_norm(self.h2, 'fc2_bn', tf.cast(self.is_training, tf.bool))
                # self.h2_dropout = tf.nn.dropout(self.h2_bn,1 - 0.05)


                self.w3 = tf.get_variable('fc3_w', [self.n_hidden_2, self.n_hidden_3], tf.float32, tf.random_normal_initializer(self.mean, self.std))
                self.b3 = tf.get_variable('fc3_b', [self.n_hidden_3], tf.float32, tf.random_normal_initializer(self.mean, self.std))

                self.h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(self.h2_bn, self.w3), self.b3)), self.relu_clip)
                self.h3_bn = self.batch_norm(self.h3, 'fc3_bn', tf.cast(self.is_training, tf.bool))
                # self.h3_dropout = tf.nn.dropout(self.h3_bn,1 - 0.05)

            with tf.name_scope('biRNN'):
                # reshape to [batchsize x time x 2*n_hidden_4]
                # h3_dropout = tf.reshape(h3_dropout, [shape_input_batch[0], -1, n_hidden_3])

                # reshape to [time x batchsize x 2*n_hidden_4]
                self.h3_bn = tf.reshape(self.h3_bn, [-1, self.shape_input_batch[0], self.n_hidden_3])

                self.forward_cell_1 = BasicLSTMCell(self.n_hidden_4, forget_bias=1.0, state_is_tuple=True)
                # forward_cell_1 = DropoutWrapper(forward_cell_1,1.0 - 0.0, 1.0 - 0.0)
                self.backward_cell_1 = BasicLSTMCell(self.n_hidden_4, forget_bias=1.0, state_is_tuple=True)
                # backward_cell_1 = DropoutWrapper(backward_cell_1, 1.0 - 0.0, 1.0 - 0.0)
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

                self.outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.forward_cell_1,
                                                             cell_bw=self.backward_cell_1,
                                                             inputs=self.h3_bn,
                                                             time_major=True,
                                                             sequence_length=self.seq_len,
                                                             dtype=tf.float32)

                self.outputs = tf.concat(self.outputs, 2)

                self.w5 = tf.get_variable('fc5_w', [self.n_hidden_3, self.n_hidden_5], tf.float32, tf.random_normal_initializer(self.mean, self.std))
                self.b5 = tf.get_variable('fc5_b', [self.n_hidden_5], tf.float32, tf.random_normal_initializer(self.mean, self.std))
                # reshape to [batchsize * timestep x num_cepstrum]
                self.h4 = tf.reshape(self.outputs, [-1, 2 * self.n_hidden_4])
                self.h5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(self.h4, self.w5), self.b5)), self.relu_clip)
                self.h5_bn = self.batch_norm(self.h5, 'fc5_bn', tf.cast(self.is_training, tf.bool))
                # self.h5_dropout = tf.nn.dropout(self.h5_bn,1.0 - 0.05)

            with tf.name_scope('logits'):
                self.w6 = tf.get_variable('fc6_w', [self.n_hidden_5, self.n_hidden_6], tf.float32, tf.random_normal_initializer(self.mean, self.std))
                self.b6 = tf.get_variable('fc6_b', [self.n_hidden_6], tf.float32, tf.random_normal_initializer(self.mean, self.std))

                self.h6 = tf.add(tf.matmul(self.h5_bn, self.w6), self.b6)
                # self.h6_bn = batch_norm(self.h6, 'fc6_bn', tf.cast(self.is_training, tf.bool))

                # reshape to [time x batchsize x n_hidden_7]
                self.logits = tf.reshape(self.h6, [-1, self.shape_input_batch[0], self.n_hidden_6])

            with tf.name_scope('decoder'):
                self.decode, self.log_prob = tf.nn.ctc_beam_search_decoder(inputs=self.logits,
                                                                 sequence_length=self.seq_len,
                                                                 merge_repeated=True)

                self.targets = tf.sparse_placeholder(tf.int32, [None, None], name="target")

            with tf.name_scope('loss'):
                self.ctc_loss = tf.nn.ctc_loss(labels=self.targets,
                                          inputs=self.logits,
                                          sequence_length=self.seq_len)

                self.avg_loss = tf.reduce_mean(self.ctc_loss)
                tf.summary.histogram("avg_loss", self.avg_loss)

            with tf.name_scope('accuracy'):
                self.distance = tf.edit_distance(tf.cast(self.decode[0], tf.int32), self.targets)
                self.ler = tf.reduce_mean(self.distance, name='label_error_rate')

            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha,
                                                   beta1=self.beta1,
                                                   beta2=self.beta2,
                                                   epsilon=self.epsilon)

                self.optimizer = self.optimizer.minimize(self.avg_loss)

            session.stop_timer()



    def restoring(self, checkpoint_dir):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
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
                session.start_timer()
                print("Restoring " + os.path.join(os.path.join(checkpoint_dir, last_checkpoint)))
                # saving model state
                self.saver = tf.train.Saver()
                self.saver.restore(sess, os.path.join(os.path.join(checkpoint_dir, last_checkpoint), 'tensorflow_1.ckpt'))
                session.stop_timer()
            else:
                tf.global_variables_initializer().run()
                session.start_timer()
                print("Saving Base Model...")
                now = session.now()
                self.saver = tf.train.Saver()
                target_checkpoint_dir = os.path.join(checkpoint_dir, 'DMC-' + now)
                os.makedirs(target_checkpoint_dir)
                save_path = self.saver.save(sess, os.path.join(target_checkpoint_dir, 'tensorflow_1.ckpt'))
                print("Base Model has been saved in " + save_path)
                session.stop_timer()

    def save(self,sess, checkpoint_dir,report):

        print("Saving ...")
        now = session.now()
        saver = tf.train.Saver()
        target_checkpoint_dir = os.path.join(checkpoint_dir, 'DMC-' + now)
        if not os.path.exists(target_checkpoint_dir):
            os.makedirs(target_checkpoint_dir)
        save_path = saver.save(sess, os.path.join(target_checkpoint_dir, 'tensorflow_1.ckpt'))
        print("Checkpoint has been saved on path : " + str(save_path))
        report.write("Checkpoint has been saved on path : " + str(save_path) + '\n')


    def feed(self, iteration, report_dir, checkpoint_dir, datasets):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            last_iteration = session.current_epoch
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

                training_shuffled_index = np.arange(len(datasets.training_dataset))
                np.random.shuffle(training_shuffled_index)

                for i in range(int(len(datasets.training_dataset) / int(self.training_batch))):
                    session.start_timer()
                    csv_training_values = []
                    print("=================================TRAINING PHASE BATCH #" + str(i) + " ITERATION AT " + str(
                        iter) + "=================================")
                    report_training.write(
                        "=================================TRAINING PHASE BATCH #" + str(i) + " ITERATION AT " + str(
                            iter) + "=================================" + '\n')
                    csv_training_values.append(i)
                    csv_training_values.append(self.learning_rate)
                    # get batch shuffled index

                    batch_i = []
                    target = []
                    for j in range(self.training_batch):
                        batch_i.append(datasets.training_dataset[training_shuffled_index[j + (i * self.training_batch)]])
                        target.append(datasets.target_training_dataset[training_shuffled_index[j + (i * self.training_batch)]])

                    batch_i = data_rep.sparse_dataset(batch_i)
                    print(batch_i.shape)
                    # batch_i = training_dataset[(i*batch):(i*batch)+batch]
                    sequence_length = np.array([batch_i.shape[1] for _ in range(self.training_batch)])

                    # target = target_training_dataset[(i*batch):(i*batch)+batch]
                    sparse_labels = data_rep.SimpleSparseTensorFrom(target)
                    feed = {
                        self.input_batch: batch_i,
                        self.seq_len: sequence_length,
                        self.targets: sparse_labels,
                        self.is_training: True,
                        self.alpha: self.learning_rate
                    }

                    loss, logg, _ = sess.run([self.avg_loss, self.decode, self.optimizer], feed)
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

                    session.stop_timer()
                    cycle_batch = int(len(datasets.training_dataset) / int(self.training_batch))
                    remaining_time = (((iteration - iter) * cycle_batch) - i) * session.duration
                    report_training.write("Elapsed time: " + str(session.duration) + '\n')
                    print("Remaining time : " + str(remaining_time))
                    report_training.write("Remaining time: " + str(remaining_time) + '\n')

                if iter > 0:

                    diff = np.array(training_losses) - np.array(training_old_losses)
                    th = diff.mean()
                    percentage = th / np.array(training_old_losses).mean() * 100
                    print("Learning performance : " + str(th))
                    report_training.write("Learning performance : " + str(th) + '\n')
                    report_training.write("Learning percentage : " + str(percentage) + '\n')

                self.save(sess,checkpoint_dir,report_training)






class Datasets:
    def __init__(self):
        self.training_dataset = []
        self.target_training_dataset = []
        self.testing_dataset = []
        self.target_testing_dataset = []

    def load_from_folder(self,training_dir, testing_dir):
        # load training dataset
        session.start_timer()
        print("Loading training dataset")
        for root, dirs, files in os.walk(training_dir, topdown=False):
            for file in files:
                if file[0] != '_':
                    print(os.path.join(training_dir, file))
                    self.target_training_dataset.append(np.load(os.path.join(training_dir, '_' + file)))
                    new_training_set = np.load(os.path.join(training_dir, file))
                    self.training_dataset.append(new_training_set)
        session.stop_timer()

        # load testing dataset
        session.start_timer()
        print("Loading testing dataset")
        for root, dirs, files in os.walk(testing_dir, topdown=False):
            for file in files:
                if file[0] != '_':
                    print(os.path.join(testing_dir, file))
                    self.target_testing_dataset.append(np.load(os.path.join(testing_dir, '_' + file)))
                    new_testing_set = np.load(os.path.join(testing_dir, file))
                    self.testing_dataset.append(new_testing_set)
        session.stop_timer()