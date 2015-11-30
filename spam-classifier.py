#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-11-24 19:52:16
# @Last Modified by:   ritesh
# @Last Modified time: 2015-11-29 20:25:25

import matplotlib.pyplot as plt
from textblob import TextBlob

import numpy as np
import csv
import pandas
import cPickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# SMS_COLLECTION = './spam-ham-dataset/smsspamcollection/SMSSpamcollection'
SMS_COLLECTION = './spam-ham-dataset/processed-email-ham-spam-collection'


"""1. Looking ino the dataset of messages..."""
messages = [line.rstrip() for line in open(SMS_COLLECTION)]
print len(messages)
for message_no, message in enumerate(messages[:10]):
    print message_no, message


messages = pandas.read_csv(SMS_COLLECTION, sep='\t', quoting=csv.QUOTE_NONE,
                           names=["label", "message"])

print messages[:10]
print messages.groupby('label').describe()

messages['length'] = messages['message'].map(lambda text: len(text))
print messages.head()

messages.hist(column='length', by='label', bins=50)


"""2. Data Preprocessing """
print "\nStart dataprocessing ... "

print "Using bag of words approach ..."
print messages.message.head()

print "split into tokens"
def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words

print messages.message.head().apply(split_into_tokens)

print "convert inot lemmas (normal form)"
def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

print messages.message.head().apply(split_into_lemmas)


"""3. Data to vectors """
print "\n Generating feature vectors "
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
print len(bow_transformer.vocabulary_)

message4 = messages['message'][3]
print message4

bow4 = bow_transformer.transform([message4])
print bow4
print bow4.shape

messages_bow = bow_transformer.transform(messages['message'])
print 'sparse matrix shape:', messages_bow.shape
print 'number of non-zeros:', messages_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

print "TF_IDF normalization"
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4

print "transform all ham/spam"
messages_tfidf = tfidf_transformer.transform(messages_bow)
print messages_tfidf.shape

spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])

print 'predicted:', spam_detector.predict(tfidf4)[0]
print 'expected:', messages.label[3]

all_predictions = spam_detector.predict(messages_tfidf)
print 'accuracy', accuracy_score(messages['label'], all_predictions)
print 'confusion matrix\n', confusion_matrix(messages['label'], all_predictions)
print '(row=expected, col=predicted)'
