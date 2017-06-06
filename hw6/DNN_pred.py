import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

out = sys.argv[1]

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

test = pd.read_csv('test.csv')
user = test['UserID'].values
movie = test['MovieID'].values

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

model = load_model('DNN.h5', custom_objects={"rmse":rmse})
prediction = model.predict([user, movie, age, occ])
'''
mean = 3.58171208604
std = 1.11689766115
prediction = (prediction * std) + mean
'''
with open(out, 'w') as ans:
	print('TestDataID,Rating', file=ans)
	for i in range(len(prediction)):
		print('%d,%f'%(i+1, prediction[i]), file=ans)