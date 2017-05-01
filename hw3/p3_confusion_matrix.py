from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

emotion_classifier = load_model('conf.h5')
np.set_printoptions(precision=2)

#training label
traindata = pd.read_csv('train.csv')
y_train = traindata.label
y_train = np.array(y_train)
#training feature
x_train = traindata.feature.str.split(' ')
x_train = x_train.tolist()
x_train = np.array(x_train)
x_train = x_train.astype('float32')
x_train /= 255
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
#valid
dev_feats = x_train[25000:26000]
te_labels = y_train[25000:26000]

predictions = emotion_classifier.predict_classes(dev_feats)
conf_mat = confusion_matrix(te_labels,predictions)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()