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
		train_x.append([1])
		for column in range(len(row)):
			train_x[nrow-1].append(float(row[column]))
		nrow += 1
	else:
		nrow += 1
nrow = 0
for row in train_x:
	train_x[nrow].append(float(row[1])**2) #age quad
	train_x[nrow].append(float(row[1])**3) #age tri
	train_x[nrow].append(float(row[1])**4) #age 4
	train_x[nrow].append(float(row[2])**2) #fnlwgt quad
	train_x[nrow].append(float(row[4])**2) #gain quad
	train_x[nrow].append(float(row[4])**3) #gain tri
	train_x[nrow].append(float(row[4])**4) #gain 4
	train_x[nrow].append(float(row[5])**2) #loss quad
	train_x[nrow].append(float(row[5])**3) #loss tri
	train_x[nrow].append(float(row[5])**4) #loss 4
	train_x[nrow].append(float(row[6])**2) #hour quad
	train_x[nrow].append(float(row[6])**3) #hour tri
	train_x[nrow].append(float(row[6])**4) #hour 4
	train_x[nrow].append(float(row[1]*row[6])) #product of age hour
	nrow += 1

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
dim = 121

#normalization
mean = np.mean(trainx, axis=0)
std = np.std(trainx, axis=0)
trainx = trainx.T
for i in range(dim):
	if std[i] != 0:
		trainx[i] = (trainx[i]-mean[i])/std[i]
trainx = trainx.T

#parameters
#-----------------------------
iteration = 2000
lr = 0.5
lamda = 1
#-----------------------------
weight = np.zeros(dim)
lamda_v = np.zeros(dim)
pregra = np.zeros(dim)
#-----------------------------
#training
for i in range(iteration):
	z = np.dot(trainx, weight)
	fwz = 1.0 / (1.0 + np.exp(-z))
	lamda_v = np.copy(weight)
	lamda_v[0] = 0.0	#let bias be zero
	gra = np.dot(trainx.T, (fwz - trainy)) + lamda*lamda_v
	pregra += gra**2
	ada = np.sqrt(pregra)
	weight -= lr*gra/ada
#checking
	if (i%100 == 0):
		count = 0
		prob = 1
		for j in range(len(fwz)):
			if fwz[j] >= 0.5 and trainy[j] == 1:
				count += 1
			if fwz[j] < 0.5 and trainy[j] == 0:
				count += 1
		print("accuracy : " + str(count/32561))

#set the testing data
test_x = []
nrow = 0
xtest = open(X_test, 'r')
test = csv.reader(xtest, delimiter=',')
for row in test:
	if nrow != 0:
		test_x.append([1])
		for column in range(len(row)):
			test_x[nrow-1].append(float(row[column]))
		nrow += 1
	else:
		nrow += 1
nrow = 0
for row in test_x:
	test_x[nrow].append(float(row[1])**2) #age quad
	test_x[nrow].append(float(row[1])**3) #age tri
	test_x[nrow].append(float(row[1])**4) #age 4
	test_x[nrow].append(float(row[2])**2) #fnlwgt quad
	test_x[nrow].append(float(row[4])**2) #gain quad
	test_x[nrow].append(float(row[4])**3) #gain tri
	test_x[nrow].append(float(row[4])**4) #gain 4
	test_x[nrow].append(float(row[5])**2) #loss quad
	test_x[nrow].append(float(row[5])**3) #loss tri
	test_x[nrow].append(float(row[5])**4) #loss 4
	test_x[nrow].append(float(row[6])**2) #hour quad
	test_x[nrow].append(float(row[6])**3) #hour tri
	test_x[nrow].append(float(row[6])**4) #hour 4
	test_x[nrow].append(float(row[1]*row[6])) #product of age hour
	nrow += 1
xtest.close()
testx = np.array(test_x)

#normalization
mean = np.mean(testx, axis=0)
std = np.std(testx, axis=0)
testx = testx.T
for i in range(dim):
	if std[i] != 0:
		testx[i] = (testx[i]-mean[i])/std[i]
testx = testx.T

#prediction
ofile = open(prediction, 'w')
ofile.write("id,label")
ofile.write('\n')
answer_z = np.dot(testx, weight)
answer_fwb = 1.0 / (1.0 + np.exp(-answer_z)) 
for i in range(len(answer_fwb)):
	ofile.write(str(i+1) + ',')
	if answer_fwb[i] >= 0.5:
		ofile.write('1')
	else:
		ofile.write('0')
	ofile.write('\n') 
ofile.close()