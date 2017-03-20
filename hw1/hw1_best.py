#!/usr/bin/env python3

import sys
import csv
import numpy as np

#read the train csv file
data_train = sys.argv[1]

data_table = []
d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

with open(data_train, 'r', encoding='Big5') as file:
	for row in csv.reader(file):
		if row[2] == "AMB_TEMP":
			for column in range(3,len(row)):
				d0.append(float(row[column]))
		if row[2] == "CH4":
			for column in range(3,len(row)):
				d1.append(float(row[column]))
		if row[2] == "CO":
			for column in range(3,len(row)):
				d2.append(float(row[column]))
		if row[2] == "NMHC":
			for column in range(3,len(row)):
				d3.append(float(row[column]))
		if row[2] == "NO":
			for column in range(3,len(row)):
				d4.append(float(row[column]))
		if row[2] == "NO2":
			for column in range(3,len(row)):
				d5.append(float(row[column]))
		if row[2] == "NOx":
			for column in range(3,len(row)):
				d6.append(float(row[column]))
		if row[2] == "O3":
			for column in range(3,len(row)):
				d7.append(float(row[column]))
		if row[2] == "PM10":
			for column in range(3,len(row)):
				d8.append(float(row[column]))
		if row[2] == "PM2.5":
			for column in range(3,len(row)):
				d9.append(float(row[column]))
		if row[2] == "RAINFALL":
			for column in range(3,len(row)):
				if row[column] == "NR":
					d10.append(0.0)
				else:
					d10.append(float(row[column]))
		if row[2] == "RH":
			for column in range(3,len(row)):
				d11.append(float(row[column]))
		if row[2] == "SO2":
			for column in range(3,len(row)):
				d12.append(float(row[column]))
		if row[2] == "THC":
			for column in range(3,len(row)):
				d13.append(float(row[column]))
		if row[2] == "WD_HR":
			for column in range(3,len(row)):
				d14.append(float(row[column]))
		if row[2] == "WIND_DIREC":
			for column in range(3,len(row)):
				d15.append(float(row[column]))
		if row[2] == "WIND_SPEED":
			for column in range(3,len(row)):
				d16.append(float(row[column]))
		if row[2] == "WS_HR":
			for column in range(3,len(row)):
				d17.append(float(row[column]))
data_table.append(d0)
data_table.append(d1)
data_table.append(d2)
data_table.append(d3)
data_table.append(d4)
data_table.append(d5)
data_table.append(d6)
data_table.append(d7)
data_table.append(d8)
data_table.append(d9)
data_table.append(d10)
data_table.append(d11)
data_table.append(d12)
data_table.append(d13)
data_table.append(d14)
data_table.append(d15)
data_table.append(d16)
data_table.append(d17)
file.close()

#for i in range(0,18):
#	mean = np.mean(data_table[i])
#	std = np.std(data_table[i])
#	data_table[i] = (data_table[i]-mean)/std

#train data
train_x = []
train_y = []
for i in range(12):
	for j in range(471):
		train_x.append([1])
		for l in range(9):
			train_x[471*i+j].append(data_table[9][480*i+j+l]) #pm2.5
		for l in range(4):
			train_x[471*i+j].append(data_table[8][480*i+j+l+2]) #pm10
		for l in range(4):
			train_x[471*i+j].append(data_table[9][480*i+j+l+2]**2) #pm2.5 quad
		#for l in range(1):
		#	train_x[471*i+j].append(data_table[9][480*i+j+l+8]**3) #pm2.5 tri
		for l in range(3):
			train_x[471*i+j].append(data_table[7][480*i+j+l+6]) #o3
		for l in range(2):
			train_x[471*i+j].append(data_table[5][480*i+j+l+7]) #no2	
		for l in range(2):
			train_x[471*i+j].append(data_table[8][480*i+j+l+4]**2) #pm10 quad
		for l in range(2):
			train_x[471*i+j].append(data_table[10][480*i+j+l+7]) #rainfall
		#for l in range(1):
		#	train_x[471*i+j].append(data_table[11][480*i+j+l+8]) #RH
		#for l in range(1):
		#	train_x[471*i+j].append(data_table[14][480*i+j+l+8]) #wd
		#for l in range(1):
		#	train_x[471*i+j].append(data_table[15][480*i+j+l+8]) #direc
		for l in range(2):
			train_x[471*i+j].append(data_table[16][480*i+j+l+7]) #speed
		#for l in range(1):
		#	train_x[471*i+j].append(data_table[17][480*i+j+l+8]) #ws
		train_y.append(data_table[9][480*i+j+9])
trainx = np.array(train_x)
trainy = np.array(train_y)

iteration = 300000
lr = 0.5
weight = np.zeros(29)
pregrad = np.zeros(29)
for i in range(iteration):
	y = np.dot(trainx, weight)
	gra = 2*np.dot(trainx.T, (y - trainy))
	pregrad += gra**2
	ada = np.sqrt(pregrad)
	weight -= lr*gra/ada
	if (i%100 == 0):
		print(np.sqrt(np.mean((trainy - y)**2)))


#test data
file_test = open(sys.argv[2], 'r', encoding='Big5')

ofile = open(sys.argv[3], 'w')
ofile.write("id,value")
ofile.write('\n')

test_table = []
for row in csv.reader(file_test):
	tmp = []
	for column in range(2, 11):
		if row[column] != "NR":
			tmp.append(float(row[column]))
		else:
			tmp.append(0.0)
	#mean = np.mean(tmp)
	#std = np.std(tmp)
	#if std != 0:
	#	tmp = (tmp-mean)/std
	test_table.append(tmp)
file_test.close()

test_x = []
for i in range(240):
	test_x.append([1])
	for j in range(0,9): #pm2.5
		test_x[i].append(test_table[18*i+9][j])
	for j in range(5,9): #pm10
		test_x[i].append(test_table[18*i+8][j])
	for j in range(5,9): #pm2.5 quad
		test_x[i].append(test_table[18*i+9][j]**2)
	#for j in range(8,9): #pm2.5 tri
	#	test_x[i].append(test_table[18*i+9][j]**3)
	for j in range(6,9): #o3
		test_x[i].append(test_table[18*i+7][j])
	for j in range(7,9): #no2
		test_x[i].append(test_table[18*i+5][j])
	for j in range(7,9): #pm10 quad
		test_x[i].append(test_table[18*i+8][j]**2)
	for j in range(7,9): #rainfall
		test_x[i].append(test_table[18*i+10][j])
	#for j in range(8,9): #RH
	#	test_x[i].append(test_table[18*i+11][j])
	#for j in range(8,9): #wd_hr
	#	test_x[i].append(test_table[18*i+14][j])
	#for j in range(8,9): #direc
	#	test_x[i].append(test_table[18*i+15][j])
	for j in range(7,9): #speed
		test_x[i].append(test_table[18*i+16][j])
	#for j in range(8,9): #ws
	#	test_x[i].append(test_table[18*i+17][j])
testx = np.array(test_x)
testy = np.dot(testx,weight)

for i in range(240):
	ofile.write("id_")
	ofile.write(str(i))
	ofile.write(",")
	ofile.write(str(testy[i]))
	ofile.write('\n')
ofile.close()