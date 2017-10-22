import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper
from timeit import default_timer as timer
import configparser as cp


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

with tf.device('/cpu:0'):
    start = timer()
    print("Building the model")
    alpha = tf.Variable(0.001, name="alpha")
    is_training = tf.placeholder(tf.bool, name="is_training")

    # initialize input network
    input_batch = tf.placeholder(tf.float32, [None, None, None], "input")
    seq_len = tf.placeholder(tf.int32, [None], name="sequence_length")

    with tf.name_scope('forward-net'):
        shape_input_batch = tf.shape(input_batch)

        # Permute n_steps and batch_size
        transpose_input_batch = tf.transpose(input_batch, [1, 0, 2])

        # reshape to [batchsize * timestep x num_cepstrum]
        reshape_input_batch = tf.reshape(transpose_input_batch, [-1, num_cep])
        print(reshape_input_batch)
        w1 = tf.get_variable('fc1_w', [num_cep, n_hidden_1], tf.float32, tf.random_normal_initializer(mean, std))
        b1 = tf.get_variable('fc1_b', [n_hidden_1], tf.float32, tf.random_normal_initializer(mean, std))

        h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(reshape_input_batch, w1), b1)), relu_clip)
        h1_bn = batch_norm(h1, 'fc1_bn', tf.cast(is_training, tf.bool))
        # h1_dropout = tf.nn.dropout(h1_bn,1 - 0.05)


        w2 = tf.get_variable('fc2_w', [n_hidden_1, n_hidden_2], tf.float32, tf.random_normal_initializer(mean, std))
        b2 = tf.get_variable('fc2_b', [n_hidden_2], tf.float32, tf.random_normal_initializer(mean, std))

        h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1_bn, w2), b2)), relu_clip)
        h2_bn = batch_norm(h2, 'fc2_bn', tf.cast(is_training, tf.bool))
        # h2_dropout = tf.nn.dropout(h2_bn,1 - 0.05)


        w3 = tf.get_variable('fc3_w', [n_hidden_2, n_hidden_3], tf.float32, tf.random_normal_initializer(mean, std))
        b3 = tf.get_variable('fc3_b', [n_hidden_3], tf.float32, tf.random_normal_initializer(mean, std))

        h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2_bn, w3), b3)), relu_clip)
        h3_bn = batch_norm(h3, 'fc3_bn', tf.cast(is_training, tf.bool))
        # h3_dropout = tf.nn.dropout(h3_bn,1 - 0.05)

    with tf.name_scope('biRNN'):
        # reshape to [batchsize x time x 2*n_hidden_4]
        # h3_dropout = tf.reshape(h3_dropout, [shape_input_batch[0], -1, n_hidden_3])

        # reshape to [time x batchsize x 2*n_hidden_4]
        h3_bn = tf.reshape(h3_bn, [-1, shape_input_batch[0], n_hidden_3])

        forward_cell_1 = BasicLSTMCell(n_hidden_4, forget_bias=1.0, state_is_tuple=True)
        # forward_cell_1 = DropoutWrapper(forward_cell_1,1.0 - 0.0, 1.0 - 0.0)
        backward_cell_1 = BasicLSTMCell(n_hidden_4, forget_bias=1.0, state_is_tuple=True)
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

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell_1,
                                                     cell_bw=backward_cell_1,
                                                     inputs=h3_bn,
                                                     time_major=True,
                                                     sequence_length=seq_len,
                                                     dtype=tf.float32)

        outputs = tf.concat(outputs, 2)

        w6 = tf.get_variable('fc6_w', [n_hidden_3, n_hidden_6], tf.float32, tf.random_normal_initializer(mean, std))
        b6 = tf.get_variable('fc6_b', [n_hidden_6], tf.float32, tf.random_normal_initializer(mean, std))
        # reshape to [batchsize * timestep x num_cepstrum]
        h5 = tf.reshape(outputs, [-1, 2 * n_hidden_5])
        h6 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h5, w6), b6)), relu_clip)
        h6_bn = batch_norm(h6, 'fc6_bn', tf.cast(is_training, tf.bool))
        # h6_dropout = tf.nn.dropout(h6_bn,1.0 - 0.05)

    with tf.name_scope('logits'):
        w7 = tf.get_variable('fc7_w', [n_hidden_6, n_hidden_7], tf.float32, tf.random_normal_initializer(mean, std))
        b7 = tf.get_variable('fc7_b', [n_hidden_7], tf.float32, tf.random_normal_initializer(mean, std))

        h7 = tf.add(tf.matmul(h6_bn, w7), b7)
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


# RUN MODEL
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:


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



