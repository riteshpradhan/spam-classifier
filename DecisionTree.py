#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-11-29 20:32:03
# @Last Modified by:   ritesh
# @Last Modified time: 2015-11-30 21:23:08

# http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html

import os
import subprocess
from time import time
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedKFold
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from textblob import TextBlob

from FeatureVector import FeatureVector
from pylib import plot_learning_curve

# SMS_COLLECTION = './spam-ham-dataset/smsspamcollection/SMSSpamcollection'
SMS_COLLECTION = './spam-ham-dataset/processed-email-ham-spam-collection'

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters

def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return  top_params


def visualize_tree(tree, feature_names, fn="dt"):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn Decision Tree.
    feature_names -- list of feature names.
    fn -- [string], root of filename, default `dt`.
    """
    dotfile = fn + ".dot"
    pngfile = fn + ".png"

    with open(dotfile, 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", dotfile, "-o", pngfile]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, "
             "to produce visualization")


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]



def main():
	print ("DecisionTree Approach")
	print ("Generating messages ...")
	feature_vector = FeatureVector(SMS_COLLECTION)
	feature_vector.data_process(sep='\t')
	messages = feature_vector.messages

	print "Splitting into train and cross-validation sets ..."
	msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)
	print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)
	print msg_train.shape, msg_test.shape

	print "\nCreating Pipeline for the analyzing and training ..."
	dt_old = Pipeline([
	    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
	    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
	    ('classifier', DecisionTreeClassifier(min_samples_split=20, random_state=99)),  # train on TF-IDF vectors w/ DecisionTree classifier
	])
	print("pipeline:", [name for name, _ in dt_old.steps])
	print("-- 10-fold cross-validation , without any grid search")
	dt_old.fit(msg_train, label_train)
	scores = cross_val_score(dt_old, msg_train, label_train, cv=10)
	print "mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std())

	from sklearn.externals.six import StringIO
	import pydot

	dot_data = StringIO()
	classes = ["ham","spam"]
	vocab = dt_old.named_steps['bow'].get_feature_names()
	vocab1 = [v.encode('ascii','ignore') for v in vocab]
	# print "vocab: ", vocab1
	with open("./plots/heme.dot", "w") as f:
	    export_graphviz(dt_old.named_steps['classifier'], out_file=f, max_depth=13, feature_names=vocab1)
	print("Creating a visualization of decision tree")
	# graph = pydot.graph_from_dot_data(dot_data.getvalue())
	# graph.write_pdf("./plots/heme.pdf")

	print "\nScore in 20% of test dataset"
	test_predictions = dt_old.predict(msg_test)
	print 'accuracy', accuracy_score(label_test, test_predictions)
	print 'confusion matrix\n', confusion_matrix(label_test, test_predictions)
	print '(row=expected, col=predicted)'
	print classification_report(label_test, test_predictions)



	# from IPython.display import Image
	# dot_data = StringIO()
	# tree.export_graphviz(dt_old, out_file=dot_data,
	#                          feature_names=iris.feature_names,
	#                          class_names=iris.target_names,
	#                          filled=True,
	#                          rounded=True,
	#                          special_characters=True)
	# graph = pydot.graph_from_dot_data(dot_data.getvalue())
	# Image(graph.create_png())


	# print ("\n---- With grid search option ---")
	# print("-- Grid Parameter Search via 10-fold CV")
	# # set of parameters to test
	# param_grid = {"classifier__criterion": ["gini", "entropy"],
	#               "classifier__min_samples_split": [5, 10, 20],
	#               "classifier__max_depth": [None, 2, 5, 10],
	#               "classifier__min_samples_leaf": [1, 5, 10],
	#               "classifier__max_leaf_nodes": [None, 5, 10, 20],
	#               }

	# dt = Pipeline([
	#     ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
	#     ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
	#     ('classifier', DecisionTreeClassifier()), # train on TF-IDF vectors w/ DecisionTree classifier
	# ])
	# print("pipeline:", [name for name, _ in dt.steps])
	# ts_gs = run_gridsearch(msg_train, label_train, dt, param_grid, cv=10)
	# print("\n-- Best Parameters:")
	# for k, v in ts_gs.items():
	#     print("parameter: {:<20s} setting: {}".format(k, v))


	# # test the retuned best parameters
	# print("\n\n-- Testing best parameters [Grid]...")
	# dt_ts_gs = Pipeline([
	#     ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
	#     ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
	#     ('classifier', DecisionTreeClassifier(**ts_gs)), # train on TF-IDF vectors w/ DecisionTree classifier
	# ])
	# print("pipeline:", [name for name, _ in dt_ts_gs.steps])
	# scores = cross_val_score(dt_ts_gs, msg_train, label_train, cv=10)
	# print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
	#                                           scores.std()))

	# print("\n-- get_code for best parameters [Grid]: ")
	# dt_ts_gs.fit(msg_train, label_train)


if __name__ == '__main__':
	main()