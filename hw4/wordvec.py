import word2vec
import numpy as np
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

#train = True
train = False

if train:
    # DEFINE your parameters for training
    corpus_path = 'all.txt'
    output_path = 'model.bin'
    MIN_COUNT = 10
    WORDVEC_DIM = 1000
    WINDOW = 3
    SAMPLE = 1e-5
    NEGATIVE_SAMPLES = 8
    ITERATIONS = 50
    MODEL = 1
    LEARNING_RATE = 0.025

    # train model
    word2vec.word2vec(
        train=corpus_path,
        output=output_path,
        cbow=MODEL,
        size=WORDVEC_DIM,
        sample=SAMPLE,
        min_count=MIN_COUNT,
        window=WINDOW,
        negative=NEGATIVE_SAMPLES,
        iter_=ITERATIONS,
        alpha=LEARNING_RATE,
        verbose=True)

else:
    # load model for plotting
    model = word2vec.load('model.bin')
    vocabs = []                 
    vecs = []                   
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:800]
    vocabs = vocabs[:800]

    #Dimensionality Reduction
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)

    # ploting and filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    # plt.savefig('hp.png', dpi=600)
    plt.show()