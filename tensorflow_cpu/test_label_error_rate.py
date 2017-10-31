import tensorflow as tf
import numpy as np


x = tf.SparseTensor(
    [[0, 0, 0],
     [1, 0, 0]],["a", "b"],(2, 1, 1))
target = tf.SparseTensor(
    [[0, 1, 0],
     [1, 0, 0],
     [1, 0, 1],
     [1, 1, 0]],["a", "b", "c", "a"],(2, 2, 2))
ler = tf.edit_distance(x,target)
with tf.Session() as sess:
    _ler = sess.run(ler)
    print(_ler)


