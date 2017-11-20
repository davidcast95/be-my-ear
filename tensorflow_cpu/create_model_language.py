import sys
sys.path.append("../")
import os
import json
from modules.model.language import word_to_word_gram




if (len(sys.argv) < 4):
    print("DATASET_DIR ~> dataset dir containing label with .txt files")
    print("N_GRAM ~> number of gram")
    print("TARGET_JSON ~> target json file")
else:
    dataset_dir = sys.argv[1]
    n = int(sys.argv[2])
    target_jsonfile = sys.argv[3]

    word_gram = dict()
    word_count = dict()
    dictionaries = []

    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        for file in files:
            if file[-4:] == '.txt':
                with open(os.path.join(root, file), encoding='utf-8') as filetarget:
                    target = filetarget.read()
                    target = target.replace(',','')
                    target = target.replace('.','')
                    target = target.replace('-',' ')
                    target = target.replace('?',' ')
                    words = target.split(' ')
                    for word in words:
                        word = word.replace('\n','')
                        if (not word in dictionaries):
                            dictionaries.append(word)
                            word_to_word_gram(word_gram,word,n)


    dictionaries.sort()

    with open(target_jsonfile, 'w') as outfile:
        json.dump({'dictionaries':dictionaries,'word_gram':word_gram}, outfile)