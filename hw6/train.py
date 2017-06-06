import numpy as np
import pandas as pd
from keras.layers import Embedding, Dropout, Dense, Input, Flatten, Concatenate, Dot, Add
from keras.models import load_model, Model
from keras.optimizers import Adam, Adamax
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import backend as K

epochs = 1000
batch_size = 4096
embedding_dim = 200
RNG_SEED = 1446557

def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))

#read train
train = pd.read_csv('train.csv')
max_userid = train['UserID'].drop_duplicates().max()+1
max_movieid = train['MovieID'].drop_duplicates().max()+1
shuffled_ratings = train.sample(frac=1.0, random_state=RNG_SEED)
user = shuffled_ratings['UserID'].values
movie = shuffled_ratings['MovieID'].values
rating = shuffled_ratings['Rating'].values
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
user_vec = Dropout(0.5)(user_vec)
user_bias = Embedding(max_userid, 1, embeddings_initializer='zeros')(user_input)
user_bias = Flatten()(user_bias)

movie_input = Input(shape=[1])
movie_vec = Embedding(max_movieid, embedding_dim, embeddings_initializer='random_normal')(movie_input)
movie_vec = Flatten()(movie_vec)
movie_vec = Dropout(0.5)(movie_vec)
movie_bias = Embedding(max_movieid, 1, embeddings_initializer='zeros')(movie_input)
movie_bias = Flatten()(movie_bias)

result = Dot(axes=1)([user_vec, movie_vec])
result = Add()([result, user_bias, movie_bias])

model = Model(inputs=[user_input, movie_input], outputs=result)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=[rmse])

callbacks = [EarlyStopping(monitor='val_rmse', 
			   patience=10, 
			   verbose=1, 
			   mode='min'), 
             ModelCheckpoint(filepath='model.h5',
			     verbose=1,
			     save_best_only=True,
			     monitor='val_rmse',
			     mode='min')]

model.fit([user, movie], rating, 
	  epochs=epochs, 
	  batch_size=batch_size,
	  validation_split=0.1,
	  callbacks=callbacks)