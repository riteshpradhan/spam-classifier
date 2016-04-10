#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-11-24 19:41:45
# @Last Modified by:   ritesh
# @Last Modified time: 2015-11-30 20:43:16

import numpy as np
import pandas

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from FeatureVector import FeatureVector

SMS_COLLECTION = './spam-ham-dataset/smsspamcollection/SMSSpamcollection'
# SMS_COLLECTION = './spam-ham-dataset/processed-email-ham-spam-collection'

def test_bow_transformer(messages, bow_transformer, tfidf_transformer, spam_detector):
	print "..... .... ..."
	print "Testing a message and bow_transformer..."
	message4 = messages['message'][3]
	print message4
	bow4 = bow_transformer.transform([message4])
	print bow4
	print bow4.shape
	print "word at 6736 is ", bow_transformer.get_feature_names()[6736]
	print "word at 8013 is ", bow_transformer.get_feature_names()[8013]
	print "End of Testing a message and bow_transformer..."

	tfidf4 = tfidf_transformer.transform(bow4)
	print tfidf4

	print "checking the tfidf transformer"
	print tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]
	print tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

	print 'predicted:', spam_detector.predict(tfidf4)[0]
	print 'expected:', messages.label[3]
	print "..... .... ..."


def main():
	feature_vector = FeatureVector(SMS_COLLECTION)
	feature_vector.data_process(sep='\t')
	messages = feature_vector.messages
	feature_vector.transformer()
	bow_transformer = feature_vector.bow_transformer
	messages_bow = feature_vector.messages_bow

	print "Describing the messages ..."
	print messages.groupby('label').describe()
	print 'sparse matrix shape:', messages_bow.shape
	print 'number of non-zeros:', messages_bow.nnz
	print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))


	print "TF_IDF normalization ... "
	tfidf_transformer = TfidfTransformer().fit(messages_bow)
	messages_tfidf = tfidf_transformer.transform(messages_bow)
	print "transform all ham/spam ... ", messages_tfidf.shape
	spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
	all_predictions = spam_detector.predict(messages_tfidf)
	print 'accuracy', accuracy_score(messages['label'], all_predictions)
	print 'confusion matrix\n', confusion_matrix(messages['label'], all_predictions)
	print '(row=expected, col=predicted)'
	print classification_report(messages['label'], all_predictions)

	test_bow_transformer(messages, bow_transformer, tfidf_transformer, spam_detector)


if __name__ == '__main__':
	main()

