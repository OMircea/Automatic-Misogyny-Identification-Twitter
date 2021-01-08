import nltk
import pandas as pd
import numpy as np
from collections import Counter
import random
from nltk.tokenize import TweetTokenizer
from sklearn.neighbors import KNeighborsClassifier



def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''

    return nltk.WordPunctTokenizer().tokenize(text)
    #return nltk.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(text)


def get_representation(vocabulary, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    wrd: @  che  .   ,   di  e
    idx: 0   1   2   3   4   5
    '''
    most_comm = vocabulary.most_common(how_many)
    wd2idx = {}
    idx2wd = {}

    for i, iterator in enumerate(most_comm):
        cuv = iterator[0]
        wd2idx[cuv] = i
        idx2wd[i] = cuv

    return wd2idx, idx2wd


def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus.
    '''

    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)

    return counter


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    features = features.astype(float)

    tokenz = tokenize(text)
    for tok in tokenz:
        if tok in wd2idx:
            features[wd2idx[tok]] += 1

    return features


def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''

    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))

    all_features = np.array(all_features)
    return all_features


def split():
    print(1)


def cross_validation(k, data, labels):
    chunk_size = len(labels) // k
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i + chunk_size]
        train_indici = np.concatenate([indici[0:i], indici[i + chunk_size:]])
        valid = data[valid_indici]
        train = data[train_indici]
        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid


def write_prediction(predictii):
    vect = np.arange(5001, 6001, 1)
    df = pd.DataFrame({'id': vect, 'label': predictii})
    df['label'] = df['label'].astype('int')
    df.to_csv(r'C:\Users\Mircea\Desktop\submission1_OpreaMircea.csv', index=False, header=True)


train_df0 = pd.read_csv('train.csv')
test_df0 = pd.read_csv('test.csv')
corpus_x0 = train_df0['text']
corpus_y0 = test_df0['text']

toate_cuvintele_100 = get_corpus_vocabulary(corpus_x0)
wd2idx_100, idx2wd_100 = get_representation(toate_cuvintele_100, 10)
data_x0 = corpus_to_bow(corpus_x0, wd2idx_100)
test_data = corpus_to_bow(test_df0['text'], wd2idx_100)

train_x_100 = data_x0
train_y_100 = train_df0['label']
test_x_100 = test_data


from sklearn.metrics import f1_score
import time as timp
clf = KNeighborsClassifier(n_neighbors=11)
rez =[]
cnt=1
inc = timp.time()
for train, valid, y_train, y_valid in cross_validation(10, data_x0, train_df0['label']):
    clf.fit(train, y_train)
    predictii = clf.predict(valid)
    scor_knn = f1_score(y_valid, predictii)
    print("Rezultatul ",cnt, ":", scor_knn)
    rez.append(scor_knn)
    cnt+=1
sf = timp.time()
print("Rezultat: ", np.mean(rez), ' ', np.std(rez))
print("Durata antrenarii: ", sf - inc)


clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(train_x_100, train_y_100)
predictii_knn = clf.predict(test_x_100)
# print(predictii_knn)
# write_prediction(predictii_knn)