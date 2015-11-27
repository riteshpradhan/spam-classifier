#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-11-09 12:41:30
# @Last Modified by:   ritesh
# @Last Modified time: 2015-11-26 18:49:30




import numpy as np
import csv
import pandas
import cPickle
import matplotlib.pyplot as plt

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class FeatureVector:

	def __init__(self, dataset):
		self.dataset = dataset
		self.messages = None
		self.bow_transformer = None
		self.messages_bow = None

	def data_process(self, sep='\t'):
		"""1. From dataset to messages """
		self.messages = [line.rstrip() for line in open(self.dataset)]
		self.messages = pandas.read_csv(self.dataset, sep=sep,
							quoting=csv.QUOTE_NONE, names=["label", "message"])
		self.messages['length'] = self.messages['message'].map(lambda text: len(text))

		"""2. Data Preprocessing """
		self.bow_transformer = CountVectorizer(analyzer=self.split_into_lemmas).fit(self.messages['message'])
		self.messages_bow = self.bow_transformer.transform(self.messages['message'])

	@classmethod
	def split_into_tokens(self):
	    message = unicode(self.message, 'utf8')  # convert bytes into proper unicode
	    return TextBlob(message).words

	def split_into_lemmas(self, message):
		message = unicode(message, 'utf8').lower()
		words = TextBlob(message).words
		# for each word, take its "base form" = lemma
		return [word.lemma for word in words]






