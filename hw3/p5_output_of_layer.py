import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd


emotion_classifier = load_model('final.h5')
layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

input_img = emotion_classifier.input
name_ls = ['conv2d_3']
collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

traindata = pd.read_csv('train.csv')
x_train = traindata.feature[1000].split(' ')
x_train = np.array(x_train, dtype=float)
x_train = x_train/255
x_train = x_train.reshape(1,48,48,1)
photo = x_train

for cnt, fn in enumerate(collect_layers):
    im = fn([photo, 0]) #get the output of that layer
    fig = plt.figure(figsize=(14, 8))
    nb_filter = 32
    for i in range(nb_filter):
        ax = fig.add_subplot(nb_filter/16, 16, i+1)
        ax.imshow(im[0][0, :, :, i], cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer{} (Given image{})'.format(cnt, 1000))
    fig.savefig(('./layer_output/'+str(cnt)))

