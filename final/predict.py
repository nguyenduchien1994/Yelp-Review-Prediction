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

# command line imports
import os, sys, argparse

# algo imports
# import json
# import numpy as numpy
# import scipy
import pandas as pd 
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

#--------------------------------------------------------
# Non-Command line params
#--------------------------------------------------------

TFIDF_FEATURES = 1000
MLP_NEURONS    = 100

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

#--------------------------------------------------------
# Subfunctions
#--------------------------------------------------------

def csv_proc(file, rows):
	review_info = pd.read_csv('reviews.csv', encoding='ISO-8859-1', nrows=rows)

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
	return shuffle(features, labels)

def tf_idf(info):
	vectorizer = TfidfVectorizer(stop_words='english', max_features=TFIDF_FEATURES, ngram_range=(1,2))
	return vectorizer.fit_transform(features)

def accuracy(preds, labels):
	count = 0
	total = len(labels)
	for i in range(total)
		count += 1 if (labels[i] == preds[i]) else 0
	return (float(count)/float(total))

def model_eval(classifier, x_train, x_test, y_train, y_test):
	classifier = LinearSVC()
	classifier.fit(x_train, y_train)
	preds = classifier.predict(x_test)
	print(" Accuracy || MSE\n%.4f%11.4f" % (accuracy(preds, y_test), mean_squared_error(y_test, preds)))
	
# def svm_run(x_train, x_test, y_train, y_test):
# 	classifier = LinearSVC()
# 	classifier.fit(x_train, y_train)
# 	preds = classifier.predict(x_test)
# 	print(" Accuracy || MSE\n%.4f%11.4f" % (accuracy(preds, y_test), mean_squared_error(y_test, preds)))

# def mlp_run(x_train, x_test, y_train, y_test):
# 	mlp_nn = MLPClassifier(solver='lbfgs', activation='logistic', tol=1e-4, alpha=1e-5, hidden_layer_sizes=(MLP_NEURONS))
# 	mlp.fit(x_train, y_train)
# 	preds = mlp.predict(x_test)
# 	print(" Accuracy || MSE\n%.4f%11.4f" % (accuracy(preds, y_test), mean_squared_error(y_test, preds)))