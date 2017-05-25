import numpy as np
import pickle
import keras.backend as K 
from keras.preprocessing.text import Tokenizer

#read train
f = open('train_data.csv', 'r', encoding='ISO-8859-1')
next(f, None)
tags = []
texts_train = []
for line in f.readlines():
	line_split = line.split(',')
	tag = line_split[1].strip('"')
	tag = tag.split(' ')
	tags.append(tag)
	text_str = "".join(str(x) for x in line_split[2:])
	text_str = text_str.split('\n')[0]
	texts_train.append(text_str)
f.close()

#read test
f = open('test_data.csv', 'r', encoding='ISO-8859-1')
next(f, None)
texts_test = []
for line in f.readlines():
	line_split = line.split(',')
	text_str = "".join(str(x) for x in line_split[1:])
	text_str = text_str.split('\n')[0]
	texts_test.append(text_str)
f.close()

#set token
all_corpus = texts_train + texts_test
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_corpus)
pickle.dump(tokenizer, open('tokenizer', "wb"))