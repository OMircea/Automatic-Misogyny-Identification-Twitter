import nltk
import pandas as pd
import numpy as np
from collections import Counter
import re
import random
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import naive_bayes
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import ItalianStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
import spacy
sp = spacy.load('it')

TRAIN_FILE = ''
TEST_FILE = ''
TXT_COL = 'text'
LBL_COL = 'label'


def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''
    text = text.lower() # lowercase all text
    text = re.sub(r'@[A-Z0-9a-z_:!@#$%^&()=+,.></?;|@#]+', 'user', text)  # replace users with "user"
    text = text.replace("#", "")  # delete hashtags
    text = re.sub('https?://[A-Za-z0-9./#]+', 'link', text)  # replace links with "link"
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.strip()  # remove leading and ending spaces

    res = ""
    stop_words = set(stopwords.words('italian'))
    text = text.split()
    for word in text:
        cuv = word
        for stop_word in stop_words:
            if word == stop_word or len(word) < 4: # deleting most common words in italian and words that have less than 4 characters
                cuv = ""
        res += cuv + " "
    res = res[:-1]
    result = res

    if len(result) > 2:
        if result[0] == " ":
            result = result[1:]

    result = ''.join(result)

    '''
    for cuvant in result:
        if len(cuvant) < 4:
            result = result.replace(cuvant, "")
    stemming = ''.join(result)
    

    stemming = [stemmer.stem(k) for k in result]
    stemming = ' '.join(stemming)
    stemming = sp(stem)
    lemma = []
    for cuvant in stemming:
        lemma.append(cuvant.lemma_)
    result = ' '.join(lemma)
    '''
    # return nltk.WordPunctTokenizer().tokenize(result)
    return nltk.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(result)


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

    # counter = Counter({k: counter for k, counter in counter.items() if counter>=10}) #using words that appear more than a specific number of times
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


import math
N = 5000
def tf_idf(corpus, wd2idx):
    all_features = []
    for text in corpus:
        features = np.zeros(len(wd2idx))
        features = features.astype(float)

        tokenz = tokenize(text)
        for tok in tokenz:
            if tok in wd2idx:
                features[wd2idx[tok]] += 1

        for tok in tokenz:
            if tok in wd2idx:
                if features[wd2idx[tok]] != 0:
                    features[wd2idx[tok]] = ((features[wd2idx[tok]] / (len(tokenz))).astype(float)) * math.log(
                        N / ap(tok, corpus))

        all_features.append(features)

    all_features = np.array(all_features)

    return all_features


def ap(tok, corpus):
    cnt = 0
    for text in corpus:
        if tok in text:
            cnt += 1

    if cnt == 0:
        return 1
    else:
        return cnt

    return cnt


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
    df.to_csv(r'C:\Users\Mircea\Desktop\submission20_OpreaMircea.csv', index=False, header=True)


train_df0 = pd.read_csv('train.csv')
test_df0 = pd.read_csv('test.csv')
corpus_x0 = train_df0['text']
corpus_y0 = test_df0['text']

toate_cuvintele_100 = get_corpus_vocabulary(corpus_x0)
wd2idx_100, idx2wd_100 = get_representation(toate_cuvintele_100, len(toate_cuvintele_100))

data_x0 = corpus_to_bow(corpus_x0, wd2idx_100)
# data_x0 = tf_idf(corpus_x0, wd2idx_100)

test_data = corpus_to_bow(test_df0['text'], wd2idx_100)
# test_data = tf_idf(test_df0['text'], wd2idx_100)


train_x_100 = data_x0
train_y_100 = train_df0['label']
test_x_100 = test_data

from sklearn.metrics import f1_score

'''
vect = []
for i in range(0, 100):
    X_train, X_test, Y_train, Y_test = train_test_split(train_x_100, train_y_100, test_size=0.25)
    Naivee = naive_bayes.MultinomialNB(alpha=0.5)
    Naivee.fit(X_train, Y_train)
    predictii_naive = Naivee.predict(X_test)
    vect.append(f1_score(Y_test, predictii_naive))
print(np.mean(vect), ' ', np.std(vect), ' ')
'''


import time as timp
from sklearn.metrics import confusion_matrix
clf = naive_bayes.MultinomialNB(alpha=0.5)
matrice_de_confuzie = np.zeros((2, 2))
rez =[]
cnt=1
inc = timp.time()
for train, valid, y_train, y_valid in cross_validation(10, data_x0, train_df0['label']): #10 fold cross-validation
    clf.fit(train, y_train)
    predictii = clf.predict(valid)
    scor_nb = f1_score(y_valid, predictii)
    print("Rezultatul", cnt, ":", scor_nb)
    rez.append(scor_nb)
    matrice_de_confuzie+= confusion_matrix(y_valid, predictii)
    cnt+=1
sf = timp.time()
print("Rezultat: ", np.mean(rez), ' ', np.std(rez))
print("Durata antrenarii: ", sf - inc)
print("Matricea de confuzie: ")
print(matrice_de_confuzie)


#calcularea si scrierea in fisier a predictiilor

Naive = naive_bayes.MultinomialNB(alpha=0.5)
Naive.fit(train_x_100, train_y_100)
predictii_naive = Naive.predict(test_x_100)
# write_prediction(predictii_naive)
# print(predictii_naive)


