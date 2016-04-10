#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-11-29 13:10:24
# @Last Modified by:   ritesh
# @Last Modified time: 2015-11-30 20:45:09

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline
from textblob import TextBlob

from FeatureVector import FeatureVector
from pylib import plot_learning_curve

"""dataset in csv format """
# SMS_COLLECTION = './spam-ham-dataset/smsspamcollection/SMSSpamcollection'
SMS_COLLECTION = './spam-ham-dataset/processed-email-ham-spam-collection'


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]


def main():
	print ("Naive-Bayes Approach")
	print "Generating messages ..."
	feature_vector = FeatureVector(SMS_COLLECTION)
	feature_vector.data_process(sep='\t')
	messages = feature_vector.messages

	print "Splitting into train and cross-validation sets ..."
	msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)
	print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)
	print msg_train.shape, msg_test.shape

	print "Creating Pipeline for the analyzing and training ..."
	pipeline = Pipeline([
	    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
	    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
	    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
	])
	print (pipeline)
	curve = plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)
	curve.savefig("./plots/acc-vs-trainSize_naive.png")
	pipeline.fit(msg_train, label_train)  #trained here

	print "Score in 20% of test dataset"
	test_predictions = pipeline.predict(msg_test)
	print 'accuracy', accuracy_score(label_test, test_predictions)
	print 'confusion matrix\n', confusion_matrix(label_test, test_predictions)
	print '(row=expected, col=predicted)'
	print classification_report(label_test, test_predictions)



if __name__ == '__main__':
	main()
