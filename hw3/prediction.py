import sys
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

#image dim
img_rows = 48
img_cols = 48

#sys.argv
test = sys.argv[1]
out = sys.argv[2]

#read test data 
testdata = pd.read_csv(test)
x_test = testdata.feature.str.split(' ')
x_test = x_test.tolist()
x_test = np.array(x_test)
x_test = x_test.astype('float32')
x_test /= 255
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#load model
model = load_model('final.h5')

#predict
prediction = model.predict_classes(x_test)
result = open(out, 'w')
result.write("id,label")
result.write("\n")
for i in range(len(prediction)):
	result.write(str(i) + "," + str(prediction[i]))
	result.write("\n")
result.close() 