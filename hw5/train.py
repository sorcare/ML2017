import numpy as np
import pickle
from keras import backend as K
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Dropout, Activation, Flatten
from keras.layers import GRU, LSTM, Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import matthews_corrcoef

embedding_dim = 200
maxlen = 306
batch_size = 128
epochs = 1000

###########################function###########################
def get_embedding_dict(path):
    embedding_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)
###########################function###########################

#read training data
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

#tags to vec
dic = ["SCIENCE-FICTION", "SPECULATIVE-FICTION", "FICTION", "NOVEL", "FANTASY", "CHILDREN'S-LITERATURE", "SATIRE", "HUMOUR", "HISTORICAL-FICTION",
	   "HISTORY", "MYSTERY", "SUSPENSE", "ADVENTURE-NOVEL", "SPY-FICTION", "AUTOBIOGRAPHY", "THRILLER", "HORROR", "ROMANCE-NOVEL", "COMEDY",
	   "NOVELLA", "WAR-NOVEL", "DYSTOPIA", "COMIC-NOVEL", "DETECTIVE-FICTION", "HISTORICAL-NOVEL", "BIOGRAPHY", "MEMOIR", "NON-FICTION",
	   "CRIME-FICTION", "AUTOBIOGRAPHICAL-NOVEL", "ALTERNATE-HISTORY", "TECHNO-THRILLER", "UTOPIAN-AND-DYSTOPIAN-FICTION", "YOUNG-ADULT-LITERATURE",
	   "SHORT-STORY", "GOTHIC-FICTION", "APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION", "HIGH-FANTASY"]
mlb = MultiLabelBinarizer(classes=dic)
y_train = mlb.fit_transform(tags)

#text to seq
tokenizer = pickle.load(open('tokenizer', "rb"))
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts_train)
x_train = pad_sequences(sequences, maxlen=maxlen)
#get mebedding matrix from glove
embedding_dict = get_embedding_dict('./glove/glove.6B.200d.txt')
num_words = len(word_index) + 1
embedding_matrix = get_embedding_matrix(word_index, embedding_dict, num_words, embedding_dim)

#split data into training set and validation set
(X_train,Y_train),(X_val,Y_val) = split_data(x_train, y_train, 0.1)

#model
model = Sequential()
model.add(Embedding(num_words,
					embedding_dim,
					weights=[embedding_matrix],
					input_length=maxlen,
					trainable=False))

model.add(GRU(256, activation='tanh', dropout=0.5, return_sequences=True))
model.add(GRU(256, activation='tanh', dropout=0.5))

model.add(Dense(256))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(38))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='categorical_crossentropy',
			  optimizer='rmsprop',
			  metrics=[f1_score])

earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath='final.h5',
							 verbose=1,
							 save_best_only=True,
							 monitor='val_f1_score',
							 mode='max')
model.fit(X_train, Y_train,
		  validation_data=(X_val, Y_val),
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[earlystopping,checkpoint])

out = model.predict(X_val)
out = np.array(out)
y_test = np.array(Y_val)
threshold = np.arange(0.5,0.7,0.1)

acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append(matthews_corrcoef(y_test[:,i],y_pred))
    acc = np.array(acc)
    index = np.where(acc==acc.max()) 
    accuracies.append(acc.max()) 
    best_threshold[i] = threshold[index[0][0]]
    acc = []
print(best_threshold)
pickle.dump(best_threshold, open('best_threshold_final', "wb"))