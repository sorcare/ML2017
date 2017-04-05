#!/usr/bin/env python3

import sys
import numpy as np
import csv

X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test = sys.argv[5]
prediction = sys.argv[6]

#set the training data
train_x = []
nrow = 0
xtrain = open(X_train, 'r')
xdata = csv.reader(xtrain, delimiter=',')
for row in xdata:
	if nrow != 0:
		train_x.append([])
		for column in range(len(row)):
			train_x[nrow-1].append(float(row[column]))
		nrow += 1
	else:
		nrow += 1
nrow = 0

train_y = []
ytrain = open(Y_train, 'r')
ydata = csv.reader(ytrain)
for row in ydata:
	train_y.append(float(row[0]))

xtrain.close()
ytrain.close()
trainx = np.array(train_x)
trainy = np.array(train_y)

#feature dim
dim = 106

#calculate mean1 and mean2
mean1 = np.zeros(dim)
mean2 = np.zeros(dim)
count1 = 0
count2 = 0
for i in range(32561):
	if trainy[i] == 1:
		mean1 += trainx[i]
		count1 += 1
	else:
		mean2 += trainx[i]
		count2 += 1
mean1 = mean1/count1
mean2 = mean2/count2

#calculate sigma1 sigma2 and share_sigma
sigma1 = np.zeros((dim, dim))
sigma2 = np.zeros((dim, dim))
for i in range(32561):
	if trainy[i] == 1:
		sigma1 += np.dot(np.transpose([trainx[i]-mean1]), [trainx[i]-mean1])
	else:
		sigma2 += np.dot(np.transpose([trainx[i]-mean2]), [trainx[i]-mean2])
sigma1 = sigma1/count1
sigma2 = sigma2/count2
share_sigma = (float(count1)/32561)*sigma1 + (float(count2)/32561)*sigma2

#set the testing data
test_x = []
nrow = 0
xtest = open(X_test, 'r')
test = csv.reader(xtest, delimiter=',')
for row in test:
	if nrow != 0:
		test_x.append([])
		for column in range(len(row)):
			test_x[nrow-1].append(float(row[column]))
		nrow += 1
	else:
		nrow += 1

xtest.close()
testx = np.array(test_x)

#sigmoid
def sigmoid(z):
	result = 1.0 / (1.0 + np.exp(-z))
	return result

#prediction
def predict(testx, mean1, mean2, share_sigma, count1, count2):
	inverse_sigma = np.linalg.inv(share_sigma)
	w = np.dot((mean1-mean2), inverse_sigma)
	x = testx.T
	b = (-0.5)*np.dot(np.dot([mean1], inverse_sigma), mean1) \
		+ (0.5)*np.dot(np.dot([mean2], inverse_sigma), mean2) + np.log(float(count1)/count2)
	z = np.dot(w, x) + b
	y = sigmoid(z)
	return y

#outfile
ofile = open(prediction, 'w')
ofile.write("id,label")
ofile.write('\n')
answer = predict(testx, mean1, mean2, share_sigma, count1, count2)
for i in range(len(answer)):
	ofile.write(str(i+1) + ',')
	if answer[i] >= 0.5:
		ofile.write('1')
	else:
		ofile.write('0')
	ofile.write('\n') 
ofile.close()