import tensorflow as tf
import numpy as np


x = tf.SparseTensor(
    [[0, 0],
     [0, 1],
     [0, 2],
     [0, 3],
     [0, 4],
     [1, 0],
     [1, 1],
     [1, 2],
     [1, 3],
     [1, 4],],["s","i","a","l","u",
               "s","i","a","l","u"],(1, 5))

target = tf.SparseTensor(
    [[0, 0],
     [0, 1],
     [0, 2],
     [0, 3],
     [0, 4],
     [0, 5],
     [1, 0],
     [1, 1],
     [1, 2],
     [1, 3],
     [1, 4],
     [1, 5]],["h","a","n","d","a","l",
               "s","y","a","l","o","m"],(2, 6))
ler = tf.edit_distance(x,target)
with tf.Session() as sess:
    _ler = sess.run(ler)
    print(_ler)


