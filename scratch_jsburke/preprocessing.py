#!/usr/bin/env python
#------------------------------------------------
#
#------------------------------------------------
#
#	-h --help			Display this message
#
#	-q --quiet 			supress output  					(default quiet)
#	-v --verbose 		chatty
#
#	-b --businesses 	set number of businesses tracked    (default 500)
#	-r --reviews 		set number of reviews tracked 	    (default 50,000)
#	-t --train 			training set size 					(default 40,000)
#
	

import json
import numpy as numpy
import scipy
import pandas as pd 
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os, sys, argparse

#--------------------------------------------------------
# Command Line
#--------------------------------------------------------

class parse_defined_error(argparse.ArgumentParser):
	def error(self, msg = ""):
		if(msg): print("\nERROR: %s\n" % msg)
		file = open(sys.argv[0])
		for (lineNum, line) in enumerate(file):
			if(line[0] != "#"): sys.exit(msg != "")
			if((lineNum == 2) or (lineNum >= 4)): print(line[1:].rstrip("\n"))

def parse_cmd():
	parser = parse_defined_error(add_help = False)

	# Normal Args

	parser.add_argument("-h", "--help", action = "store_true")

	#maybe implement later
	# Quiet <--> Verbose
	response = parser.add_mutually_exclusive_group()

	response.add_argument("-v", "--verbose", action = "store_true")
	response.add_argument("-q", "--quiet",   action = "store_true")

	# Additional Args

	# parser.add_argument("-b", "--businesses",  type = int)
	parser.add_argument("-r", "--reviews",  type = int)
	parser.add_argument("-t", "--train",  type = int)

	parser.add_argument("-w", "--write",  type = str)

	options = parser.parse_args()
	if options.help:
		parser.error()
		sys.exit()
	return options

#-------------------------------------------------------
#  code we care about
#-------------------------------------------------------

def main():

	# business_count = 500
	reviews_count  = 50000
	ids = []
	# business_info = []
	# wrt_csv = 1
	train_set_size = 40000
	verbose = 0

	options=parse_cmd()

	# if (options.businesses):
	# 	business_count = options.businesses

	if (options.reviews):
		reviews_count = options.reviews	

	if (options.train):
		train_set_size = options.train

	# if (options.write == 'on'):
	# 	wrt_csv = 1

	if options.verbose:
		verbose = 1

	

	review_info = pd.read_csv('../reviews.csv', nrows=reviews_count, encoding='ISO-8859-1')

	if verbose == 1:
		print("file read done")

	# Create balanced distribution of samples
	d = {}
	labels = []
	features = []
	for i in range(review_info.shape[0]):
	    if review_info.rating[i] in d and d[review_info.rating[i]] < 10000:
	        d[review_info.rating[i]] += 1
	        labels.append(review_info.rating[i])
	        features.append(review_info.review[i])
	    elif review_info.rating[i] not in d:
	        d[review_info.rating[i]] = 0        

	# Shuffle lists
	features, labels = shuffle(features, labels)

	if verbose==1:
		print('features and labels set')

	vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2))
	#dtm = vectorizer.fit_transform(review_info.review.values).toarray()
	dtm = vectorizer.fit_transform(features)

	if verbose == 1:
		print("tfidf done")

	# X = dtm
	# le = preprocessing.LabelEncoder()
	# y = le.fit_transform(review_info.rating.values)
	# X_train = X[:train_set_size]
	# y_train = y[:train_set_size]
	# X_test = X[train_set_size:]
	# y_test = y[train_set_size:]

	# Use ~80% of data for training and ~20% for testing
	# X = dtm
	# y = labels
	X_train, X_test, y_train, y_test = train_test_split(dtm, labels, test_size=0.20)



	if verbose == 1:
		print("data sets established")

	preds = []

	mlp_nn = MLPClassifier(solver='lbfgs', activation='logistic', tol=1e-4, alpha=1e-5, hidden_layer_sizes=(100))

	mlp_nn.fit(X_train, y_train)

	if verbose == 1:
		print("NN trained")

	# print("mlp accuracy : %.4f" % mlp_nn.score(X_test, y_test))

	preds = mlp_nn.predict(X_test)

	count = 0
	total = len(preds)
	for i in range(total):
		count += 1 if (y_test[i] == preds[i]) else 0

	print("Accuracy || MSE \n %.4f%12.4f" % ((float(count)/float(total)) , mean_squared_error(y_test, preds)))

	# good = 0
	# count = 0

	# print('predict -- real')
	# for i in range(len(preds)-1):
	# 	count += 1
	# 	# print(preds[i+1] + '  --  ' + y_test[i+1])
	# 	if preds[i+1] == y_test[i+1]:
	# 		# print('good')
	# 		good += 1
	# 	# else:
	# 	# 	print('XXX')

	# print('good  : %d'%good)z
	# print('total : %d'%count)


	# print(X[11])
	# print(y[11])

	# print(X[42])
	# print(y[42])

main()