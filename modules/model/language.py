import tensorflow as tf

def add_gram_to_word_gram(word_gram, gram, new_word):
    if gram not in word_gram:
        word_gram[gram] = []
    word_gram[gram].append(new_word)
    # if gram not in word_gram:
    #     word_gram[gram] = dict()
    # count_index = str(len(new_word))
    # if count_index not in word_gram[gram]:
    #     word_gram[gram][count_index] = []
    # word_gram[gram][count_index].append(new_word)

def n_gram(word,n):
    gram = []
    end = len(word)
    for i in range(end):
        if len(word[i:(i+n)]) == n:
            gram.append(word[i:(i+n)])
            gram.sort()
    return gram

def word_to_word_gram(word_gram, new_word,n):
    for g in n_gram(new_word,n):
        add_gram_to_word_gram(word_gram,g,new_word)


def duplicateWordSparseTensor(word, n):
    max_len = len(word)
    indices = []
    values = []
    for i in range(n):
        for j,char in enumerate(word):
            indices.append([i,j])
            values.append(char)

    return tf.SparseTensor(indices,values,(n,max_len))

def wordSparseTensor(words, guess_words):
    max_len = len(max(words,key=len))
    utterances = len(words)
    indices = []
    values = []

    target_indices = []
    target_values = []
    for i, word in enumerate(words):

        if len(word) > len(guess_words):
            for j,char in enumerate(word):
                indices.append([i,j])
                values.append(char)

                if j < len(guess_words):
                    target_indices.append([i,j])
                    target_values.append(guess_words[j])
        else:
            for j, char in enumerate(guess_words):
                target_indices.append([i, j])
                target_values.append(char)

                if j < len(word):
                    indices.append([i, j])
                    values.append(word[j])





    return tf.SparseTensorValue(indices,values,(utterances,max_len)), tf.SparseTensorValue(target_indices,target_values,(utterances,len(guess_words)))

def create_sparse_vec(word_list):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi,x in enumerate(word_list) for yi,y in enumerate(x)]
    chars = list(''.join(word_list))
    return tf.SparseTensorValue(indices, chars, [num_words,1,1])