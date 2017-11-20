import sys
sys.path.append("../")
import tensorflow as tf
import modules.model.language as lang
import json
from timeit import default_timer as timer
import os

# if len(sys.argv) < 3:
#     print("JSON_DIR ~> location of model language as json")
#     print("INPUT_WORD ~> word produce by network")
# else:
json_dir = '//Users/davidwibisono/be-my-ear/datasets/noise_human/model.json'
w = 'meuagaku'

if not os.path.exists(json_dir):
    print("500")
else:
    model_lang = json.load(open(json_dir))

    count_words = []
    prob_words = []

    gram = lang.n_gram(w, 2)
    for g in gram:
        # sparse_word = lang.wordSparseTensor(model_lang['word_gram'][g], test_word)
        if g in model_lang['word_gram']:
            for prob_word in model_lang['word_gram'][g]:
                if prob_word in prob_words:
                    index = prob_words.index(prob_word)
                    count_words[index]+=1
                else:
                    count_words.append(1)
                    prob_words.append(prob_word)


    sparse_prob_word, sparse_input_word = lang.wordSparseTensor(prob_words,w)
    sim = tf.edit_distance(sparse_input_word, sparse_prob_word)

    with tf.Session() as sess:
        start = timer()
        res = sess.run(sim)
        for i, v in enumerate(res):
            print(str(prob_words[i]) + ' : ' + str(v) + ' / ' + str(count_words[i]) + ' = ' + str(v/count_words[i]))
        res /= count_words

        print(prob_words[res.argmin()])





