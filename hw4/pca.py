import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#load data
data = []
for i in range(10):
	filename = "./p1_data/A" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/B" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/C" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/D" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/E" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/F" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/G" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/H" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/I" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
for i in range(10):
	filename = "./p1_data/J" + "0" + str(i) + ".bmp"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(4096)
	p = p.tolist()
	data.append(p)
data = np.array(data)

#1.
data_mean = data.mean(axis=0, keepdims=True)
data_ctr = data - data_mean
u, s, v = np.linalg.svd(data_ctr)
data_mean = data_mean.reshape(64, 64)
eigen = v[0:9]
eigen = eigen.reshape(9, 64, 64)

#plot and save for 1.
plt.figure()
plt.imshow(data_mean, cmap='gray')
plt.savefig("./p1_pic/average")

fig = plt.figure()
for i in range(9):
	ax = fig.add_subplot(3, 3, i+1)
	ax.imshow(eigen[i], cmap='gray')
fig.savefig("./p1_pic/eigen")


#2.
data_mean = data_mean.reshape(4096)
eigen = eigen.reshape(9, 4096)
eigen = eigen[0:5]
x_reduce = np.dot(data_ctr, eigen.T)
x_rec = np.dot(x_reduce, eigen) + data_mean
data = data.reshape(100, 64, 64)
x_rec = x_rec.reshape(100, 64, 64)

#plot and save for 2.
fig = plt.figure()
for i in range(100):
	ax = fig.add_subplot(10, 10, i+1)
	ax.imshow(data[i], cmap='gray')
	plt.axis('off')
fig.savefig("./p1_pic/original")

fig = plt.figure()
for i in range(100):
	ax = fig.add_subplot(10, 10, i+1)
	ax.imshow(x_rec[i], cmap='gray')
	plt.axis('off')
fig.savefig("./p1_pic/recovered")

#3.
data = data.reshape(100, 4096)
dim = 0
for k in range(100):
	eigen = v[0:k+1]
	x_reduce = np.dot(data_ctr, eigen.T)
	x_rec = np.dot(x_reduce, eigen) + data_mean
	error = data - x_rec
	rmse = ((error**2).mean())**0.5 / 255
	if rmse < 0.01:
		dim = k+1
		break
print(dim)