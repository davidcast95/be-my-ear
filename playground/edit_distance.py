import tensorflow as tf

a = tf.SparseTensor([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0]],["k","i","t","t","e","n"],(6,1,1))
b = tf.SparseTensor([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0],[6,0,0]],["s","i","t","t","i","n","g"],(7,1,1))

edit = tf.edit_distance(a,b)
edit_mean = tf.reduce_mean(edit)


with tf.Session() as sess:
    e = sess.run(edit)
    print(e)
    mean = sess.run(edit_mean)
    print(mean)