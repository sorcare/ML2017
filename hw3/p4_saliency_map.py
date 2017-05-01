from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
traindata = pd.read_csv('train.csv')
x_train = traindata.feature.str.split(' ')
x_train = x_train.tolist()
x_train = np.array(x_train, dtype=float)
y_train = np.array(traindata['label'])
x_train = x_train.reshape(x_train.shape[0],48,48,1)

model = load_model('final.h5')
input_img = model.input
img_ids = [10000,10001,10002,10003,10004,10005]

for idx in img_ids:
    origin = x_train[idx].reshape(48, 48)

    plt.figure()
    plt.imshow(origin,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('./pic/'+str(idx)+".png", dpi=100)

    val_proba = model.predict((x_train[idx]/255).reshape(1,48,48,1))
    pred = val_proba.argmax(axis=-1)
    target = K.mean(model.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])

    heatmap = fn([(x_train[idx]/255).reshape(1,48,48,1),0])
    heatmap = heatmap[0]
    heatmap = np.abs(heatmap)
    heatmap -= heatmap.mean()
    heatmap /= (heatmap.std()+1e-10)
    heatmap *= 0.1
    heatmap += 0.5
    heatmap = np.clip(heatmap,0,1)
    heatmap /= np.max(heatmap)
    '''
    Implement your heatmap processing here
    hint: Do some normalization or smoothening on grads
    '''

    thres = 0.5
    see = x_train[idx].reshape(48, 48)
    heatmap = heatmap.reshape(48,48)
    see[np.where(heatmap <= thres)] = np.mean(see)

    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig("./pic/"+str(idx)+"_heatmap.png",dpi=100)

    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('./pic/'+str(idx)+"_mask.png", dpi=100)

