import sys
sys.path.append("../")
import tensorflow as tf
import modules.model.language as lang
import json
from timeit import default_timer as timer
import os

if len(sys.argv) < 3:
    print("JSON_DIR ~> location of model language as json")
    print("INPUT_SENTENCE ~> word produce by network")
else:
    json_dir = sys.argv[1]
    s = sys.argv[2]

    if not os.path.exists(json_dir):
        print("500")
    else:
        model_lang = json.load(open(json_dir))

        raw = tf.sparse_placeholder(tf.string)
        ref = tf.sparse_placeholder(tf.string)

        sim = tf.edit_distance(raw, ref)


        with tf.Session() as sess:
            sentence = ""
            words = s.split(' ')
            for w in words:
                if len(w) > 1:
                    count_words = []
                    prob_words = []

                    gram = lang.n_gram(w, 2)
                    for g in gram:
                        # sparse_word = lang.wordSparseTensor(model_lang['word_gram'][g], test_word)
                        if g in model_lang['word_gram']:
                            for prob_word in model_lang['word_gram'][g]:
                                if prob_word in prob_words:
                                    index = prob_words.index(prob_word)
                                    count_words[index] += 1
                                else:
                                    count_words.append(1)
                                    prob_words.append(prob_word)

                    sparse_prob_word, sparse_input_word = lang.wordSparseTensor(prob_words, w)
                    res = sess.run(sim, feed_dict={
                        ref : sparse_prob_word,
                        raw : sparse_input_word
                    })
                    print(prob_words[res.argmin()])




