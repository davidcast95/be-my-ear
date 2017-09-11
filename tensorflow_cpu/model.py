import os
import sys
import csv
sys.path.append("../")
from time import gmtime, strftime
import modules.features.data_representation as data_rep
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

csv_fields = ['iteration','batch','learning_rate','ctc_loss','decoded_text']

if len(sys.argv) < 4:
    print ('this method needs 3 args TRAINING_DIR CHECKPOINT_DIR REPORT_DIR')
    print ('TRAINING_DIR ~> directory of target preprocessing files will be created by preprocessing.py')
    print ("CHECKPOINT_DIR ~> directory of model's checkpoint will be stored")
    print ('REPORT_DIR ~> dicretory of result per checkpoint')
else:


    #init
    training_dir = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    report_dir = sys.argv[3]
    iteration = 200
    batch = 1
    num_cep = 129

    #property of weight
    mean = 0
    std = 0.3
    relu_clip = 20
    n_hidden_1 = 128
    n_hidden_2 = 128
    n_hidden_3 = 2 * 128
    n_hidden_4 = 128
    n_hidden_5 = 128
    n_hidden_6 = 28

    #property of BiRRN LSTM
    n_hidden_unit = 8 * 128
    forget_bias = 0

    #property of AdamOptimizer (http://arxiv.org/abs/1412.6980) parameters
    beta1 = 0.9
    beta2 = 0.9
    epsilon = 1e-6
    learning_rate = 0.001
    threshold = 0

    training_dataset = []
    target_dataset = []

    # load training dataset
    for root, dirs, files in os.walk(training_dir, topdown=False):
        for file in files:
            if file[0] == '_':
                target_dataset.append(np.load(os.path.join(training_dir, file)))
            else:
                training_dataset.append(np.load(os.path.join(training_dir, file)))

    training_dataset = data_rep.sparse_dataset(training_dataset)

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

        with tf.name_scope('fc5'):
            w5 = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_5],mean,std,tf.float64),name='fc5_w')
            b5 = tf.Variable(tf.random_normal([n_hidden_5],mean,std,tf.float64),name='fc5_b')

        with tf.name_scope('logits'):
            w6 = tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6],mean,std,tf.float64),name='logits_w')
            b6 = tf.Variable(tf.random_normal([n_hidden_6],mean,std,tf.float64),name='logits_b')




    #SETUP NETWORK
    input_training = tf.placeholder(tf.float64, [None, None, None], "input")

    seq_len = tf.placeholder(tf.int32, [None], name="sequence_length")

    # reshape to [batchsize * timestep x num_cepstrum]
    training_batch = tf.reshape(input_training, [-1, num_cep])

    #feed forward
    h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(training_batch,w1), b1)),relu_clip)
    h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1,w2), b2)),relu_clip)
    h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2,w3), b3)),relu_clip)

    # reshape to [time x batchsize x 2*n_hidden_4]
    h3 = tf.reshape(h3, [-1, batch, n_hidden_3])

    forward_cell = BasicLSTMCell(n_hidden_4, forget_bias, True)
    backward_cell = BasicLSTMCell(n_hidden_4, forget_bias, True)

    #BiRNN
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                        cell_bw=backward_cell,
                                                        inputs=h3,
                                                        dtype=tf.float64,
                                                        time_major=True,
                                                        sequence_length=seq_len)
    outputs = tf.concat(outputs, 2)

    #reshape to [batchsize * timestep x num_cepstrum]
    h4 = tf.reshape(outputs,[-1,2 * n_hidden_4])


    #fully connected
    h5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h4,w5), b5)),relu_clip)


    h6 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h5,w6), b6)),relu_clip)


    #reshape to [time x batchsize x 2*n_hidden_4]
    logits = tf.reshape(h6, [-1, batch, n_hidden_6])
    logits = tf.cast(logits,tf.float32)


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


    #RUN MODEL
    with tf.Session() as sess:
        last_iteration = 0

        valid_dirs = []
        for root, dirs, files in os.walk(checkpoint_dir, topdown=True):
            for dir in dirs:
                if dir[0] == 'D' and dir[1] == 'M' and dir[2] == 'C' and dir[3] == '-':
                    valid_dirs.append(dir)
            if len(valid_dirs) > 0:
                break
        if len(valid_dirs) > 0:
            last_iteration = len(valid_dirs)
            last_checkpoint = valid_dirs[last_iteration-1]
            print ("Restoring")
            # saving model state
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(os.path.join(checkpoint_dir,last_checkpoint),'tensorflow_1.ckpt'))



        else:
            tf.global_variables_initializer().run()
            print("Saving Base Model...")
            saver = tf.train.Saver()
            target_checkpoint_dir = os.path.join(checkpoint_dir, 'base')
            os.makedirs(target_checkpoint_dir)
            save_path = saver.save(sess, os.path.join(target_checkpoint_dir, 'base.ckpt'))

        _, num_cep = training_dataset[0].shape

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
            report.write("iteration #"+str(iter + last_iteration) + "\n")
            csv_values = []
            csv_values.append(iter + last_iteration)
            if iter > 0:
                old_losses = losses
                losses = []
            for i in range(int(len(training_dataset) / int(batch))):
                csv_values = []
                print ("batch #"+str(i))
                report.write("batch #"+str(i) + "\n")
                csv_values.append(i)
                csv_values.append(learning_rate)
                #get batch
                batch_i = training_dataset[(i*batch):(i*batch)+batch]
                sequence_length = np.array([batch_i.shape[1] for _ in range(batch)])
                target = target_dataset[(i*batch):(i*batch)+batch]
                sparse_labels = data_rep.SimpleSparseTensorFrom(target)
                feed = {
                    input_training : batch_i,
                    seq_len : sequence_length,
                    targets : sparse_labels
                }

                logg = sess.run(decode, feed)
                print ("Encoded CTC :")
                report.write("Encoded CTC :\n")
                decode_text = data_rep.indices_to_text(logg[0][1])
                print(decode_text)
                report.write(decode_text + "\n")

                loss = sess.run(avg_loss, feed)
                print ("negative log-probability :" + str(loss))
                report.write("negative log-probability :" + str(loss) + "\n")
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
                report.write("Learning performance : " + str(th) + "\n")
                report.write("Learning percentage : " + str(percentage) + "\n")

                print ("Saving ...")
                now = strftime("%d-%m-%Y-%H-%M-%S", gmtime())
                saver = tf.train.Saver()
                target_checkpoint_dir = os.path.join(checkpoint_dir, 'DMC-'+now)
                os.makedirs(target_checkpoint_dir)
                save_path = saver.save(sess, os.path.join(target_checkpoint_dir,'tensorflow_1.ckpt'))
                print ("Checkpoint has been saved on path : " + str(save_path))
                report.write("Checkpoint has been saved on path : " + str(save_path) + "\n")

                _w1 = w1.eval(sess)
                np.save(os.path.join(target_checkpoint_dir,"w1"),_w1)
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
                print ("Saving ...")
                now = strftime("%d-%m-%Y-%H-%M-%S", gmtime())
                saver = tf.train.Saver()
                target_checkpoint_dir = os.path.join(checkpoint_dir, 'DMC-'+now)
                os.makedirs(target_checkpoint_dir)
                save_path = saver.save(sess, os.path.join(target_checkpoint_dir,'tensorflow_1.ckpt'))
                print ("Checkpoint has been saved on path : " + str(save_path))
                report.write("Checkpoint has been saved on path : " + str(save_path) + "\n")

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