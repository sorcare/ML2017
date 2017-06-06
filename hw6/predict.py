import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K

data = sys.argv[1]
out = sys.argv[2]

def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))

test_data = sys.argv[1] + 'test.csv'
test = pd.read_csv(test_data)
user = test['UserID'].values
movie = test['MovieID'].values

model1 = load_model('./ensemble/350_no_nor.h5', custom_objects={"rmse":rmse})
prediction1 = model1.predict([user, movie])

model2 = load_model('./ensemble/350_nobiasnonor.h5', custom_objects={"rmse":rmse})
prediction2 = model2.predict([user, movie])

model3 = load_model('./ensemble/400_no_nor.h5', custom_objects={"rmse":rmse})
prediction3 = model3.predict([user, movie])

model4 = load_model('./ensemble/400_nobiasnonor.h5', custom_objects={"rmse":rmse})
prediction4 = model4.predict([user, movie])

model5 = load_model('./ensemble/200_no_nor.h5', custom_objects={"rmse":rmse})
prediction5 = model5.predict([user, movie])

model6 = load_model('./ensemble/200_nobiasnonor.h5', custom_objects={"rmse":rmse})
prediction6 = model6.predict([user, movie])

prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5 + prediction6) / 6.0
'''
mean = 3.58171208604
std = 1.11689766115
prediction = (prediction * std) + mean
'''
with open(out, 'w') as ans:
	print('TestDataID,Rating', file=ans)
	for i in range(len(prediction)):
		print('%d,%f'%(i+1, prediction[i]), file=ans)