#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-11-29 15:45:01
# @Last Modified by:   ritesh
# @Last Modified time: 2015-11-29 16:05:46


import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedKFold
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from textblob import TextBlob

from FeatureVector import FeatureVector
from pylib import plot_learning_curve

SMS_COLLECTION = './spam-ham-dataset/smsspamcollection/SMSSpamcollection'


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]


def main():
	print ("SVM Approach")
	print ("Generating messages ...")
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
	    ('classifier', SVC()),  # train on TF-IDF vectors w/ Naive Bayes classifier
	])

	# pipeline parameters to automatically explore and tune
	param_svm = [
	  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
	  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
	]
	print (pipeline)

	grid_svm = GridSearchCV(
	    pipeline,  	# pipeline from above
	    param_grid=param_svm,  # parameters to tune via cross validation
	    refit=True,  # fit using all data, on the best detected classifier
	    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
	    scoring='accuracy',  # what score are we optimizing?
	    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
	)
	svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
	print svm_detector.grid_scores_
	print svm_detector.predict(["Hi mom, how are you?"])[0]
	print svm_detector.predict(["fuck you!"])[0]

	print "Score in 20% of test dataset"
	test_predictions = svm_detector.predict(msg_test)
	print 'accuracy', accuracy_score(label_test, test_predictions)
	print 'confusion matrix\n', confusion_matrix(label_test, test_predictions)
	print '(row=expected, col=predicted)'
	print classification_report(label_test, test_predictions)

	curve = plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)
	curve.savefig("./plots/acc-vs-trainSize_SVM.png")
	pipeline.fit(msg_train, label_train)  #trained here

if __name__ == '__main__':
	main()