import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors

np.random.seed(10)

def get_eigenvalues(data):
    SAMPLE = 100 # sample some points to estimate
    NEIGHBOR = 200 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals

# Train a linear SVR

npzfile = np.load('train_data.npz')
X = npzfile['X']
y = npzfile['y']

svr = SVR(C=10)
svr.fit(X, y)

#load data
data = []
for i in range(481):
	filename = "./hand/hand.seq" + str(i+1) + ".png"
	im = Image.open(filename)
	p = np.array(im)
	p = p.reshape(512*480)
	p = p.tolist()
	data.append(p)
data = np.array(data)

#pca reduction dim to 100
pca = PCA(n_components=100)
data = pca.fit_transform(data)

#predict
test_X = []
vs = get_eigenvalues(data)
test_X.append(vs)
test_X = np.array(test_X)
pred_y = svr.predict(test_X)
print(pred_y)