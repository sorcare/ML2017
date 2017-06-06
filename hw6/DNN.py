import numpy as np
import pandas as pd
from keras.layers import Embedding, Dropout, Dense, Input, Flatten, Concatenate, Dot, Add
from keras.models import load_model, Model
from keras.optimizers import Adam, Adamax
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras import backend as K

epochs = 1000
batch_size = 4096
embedding_dim = 200
RNG_SEED = 1446557

def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))

#read users info
train_user = open('users.csv', 'r')
next(train_user, None)
age_dict = {}
occ_dict = {}
for line in train_user.readlines():
	line_spilt = line.split("::")
	age_dict[line_spilt[0]] = line_spilt[2]
	occ_dict[line_spilt[0]] = line_spilt[3]

#read train
train = pd.read_csv('train.csv')
max_userid = train['UserID'].drop_duplicates().max()+1
max_movieid = train['MovieID'].drop_duplicates().max()+1
shuffled_ratings = train.sample(frac=1.0, random_state=RNG_SEED)
user = shuffled_ratings['UserID'].values
movie = shuffled_ratings['MovieID'].values
rating = shuffled_ratings['Rating'].values

#add feature (age and occ)
age = []
occ = []
for i in range(len(user)):
	age.append([int(age_dict[str(user[i])])])
	occ.append([int(occ_dict[str(user[i])])])
age = pad_sequences(age)
occ = pad_sequences(occ)
max_age = age.max()+1
max_occ = occ.max()+1
'''
#normalization
mean = np.mean(rating)
std = np.std(rating)
rating = (rating-mean)/std
print(mean)
print(std)
'''
#model
user_input = Input(shape=[1])
user_vec = Embedding(max_userid, embedding_dim, embeddings_initializer='random_normal')(user_input)
user_vec = Flatten()(user_vec)

movie_input = Input(shape=[1])
movie_vec = Embedding(max_movieid, embedding_dim, embeddings_initializer='random_normal')(movie_input)
movie_vec = Flatten()(movie_vec)

age_input = Input(shape=[1])
age_vec = Embedding(max_age, embedding_dim, embeddings_initializer='random_normal')(age_input)
age_vec = Flatten()(age_vec)

occ_input = Input(shape=[1])
occ_vec = Embedding(max_occ, embedding_dim, embeddings_initializer='random_normal')(occ_input)
occ_vec = Flatten()(occ_vec)

concat_vec = Concatenate()([user_vec, movie_vec, age_vec, occ_vec])
dnn = Dense(256, activation='relu')(concat_vec)
dnn = Dropout(0.5)(dnn)
dnn = Dense(128, activation='relu')(dnn)
dnn = Dropout(0.5)(dnn)
dnn = Dense(64, activation='relu')(dnn)
dnn = Dropout(0.5)(dnn)
result = Dense(1, activation='linear')(dnn)

model = Model(inputs=[user_input, movie_input, age_input, occ_input], outputs=result)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=[rmse])

callbacks = [EarlyStopping(monitor='val_rmse', 
			   patience=10, 
			   verbose=1, 
			   mode='min'), 
             ModelCheckpoint(filepath='DNN.h5',
			     verbose=1,
			     save_best_only=True,
			     monitor='val_rmse',
			     mode='min')]

model.fit([user, movie, age, occ], rating, 
	  epochs=epochs, 
	  batch_size=batch_size,
	  validation_split=0.1,
	  callbacks=callbacks)