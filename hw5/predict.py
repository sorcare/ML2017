import sys
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import backend as K

maxlen = 306

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#sys.argv
test = sys.argv[1]
outfile = sys.argv[2]

#read test
f = open(test, 'r', encoding='ISO-8859-1')
next(f, None)
texts_test = []
for line in f.readlines():
	line_split = line.split(',')
	text_str = "".join(str(x) for x in line_split[1:])
	text_str = text_str.split('\n')[0]
	texts_test.append(text_str)
f.close()

tokenizer = pickle.load(open('tokenizer', "rb"))
sequences = tokenizer.texts_to_sequences(texts_test)
x_test = pad_sequences(sequences, maxlen=maxlen)

#load model
model = load_model('final.h5', custom_objects={"f1_score":f1_score})

#predict
dic = ["SCIENCE-FICTION", "SPECULATIVE-FICTION", "FICTION", "NOVEL", "FANTASY", "CHILDREN'S-LITERATURE", "SATIRE", "HUMOUR", "HISTORICAL-FICTION",
	   "HISTORY", "MYSTERY", "SUSPENSE", "ADVENTURE-NOVEL", "SPY-FICTION", "AUTOBIOGRAPHY", "THRILLER", "HORROR", "ROMANCE-NOVEL", "COMEDY",
	   "NOVELLA", "WAR-NOVEL", "DYSTOPIA", "COMIC-NOVEL", "DETECTIVE-FICTION", "HISTORICAL-NOVEL", "BIOGRAPHY", "MEMOIR", "NON-FICTION",
	   "CRIME-FICTION", "AUTOBIOGRAPHICAL-NOVEL", "ALTERNATE-HISTORY", "TECHNO-THRILLER", "UTOPIAN-AND-DYSTOPIAN-FICTION", "YOUNG-ADULT-LITERATURE",
	   "SHORT-STORY", "GOTHIC-FICTION", "APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION", "HIGH-FANTASY"]
prediction = model.predict(x_test)

out = np.array(prediction)
best_threshold = pickle.load(open('best_threshold_final', "rb"))

y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(out.shape[1])] for i in range(len(out))])

result = open(outfile, 'w')
result.write('"id","tags"\n')
for i in range(len(y_pred)):
	result.write('"' + str(i) + '","')
	tag_list = []
	for j in range(len(y_pred[i])):
		if y_pred[i][j] == 1:
			tag_list.append(dic[j])
	ans = " ".join(str(x) for x in tag_list)
	result.write(ans + '"\n')
result.close()