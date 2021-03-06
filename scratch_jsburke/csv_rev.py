#!/usr/bin/env python
#------------------------------------------------
#
#------------------------------------------------
#
#	-h --help			Display this message
#
#	-q --quiet 			supress output
#	-v --verbose 		chatty
#
#	-b --businesses 	set number of businesses tracked    (default 500)
#	-r --reviews 		set number of reviews tracked 	    (default 500,000)
#   -w --write			writes output to file {'on', 'off'} (default off)
#
#   It seems 500 times more reviews picked than businesses gets a good coverage
#
#	

import json
import numpy as numpy
import scipy
import pandas as pd 
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

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

	parser.add_argument("-b", "--businesses",  type = int)
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

	business_count = 500
	reviews_count  = 50000
	ids = []
	business_info = []
	wrt_csv = 1
	train_set_size = 40000
	verbose = 0

	options=parse_cmd()

	if (options.businesses):
		business_count = options.businesses

	if (options.reviews):
		reviews_count = options.reviews	

	if (options.train):
		train_set_size = options.train

	if (options.write == 'on'):
		wrt_csv = 1

	if options.verbose:
		verbose = 1

	# cities = ['Pittsburgh', 'Charlotte', 'Phoenix', 'Urbana']

	# count = 0
	# with open("../yelp_academic_dataset_business.json") as business_json:
	# 	for line in business_json:
	# 		business = json.loads(line)
	# 		if business['categories'] is not None:
	# 			if business['city'] in cities:
	# 				if 'Food' in business['categories']:
	# 					business_info.append({'business_id': business['business_id'], 'rating': business['stars']})
	# 					ids.append(business['business_id'])
	# 					count += 1
	# 				elif 'Restaurants' in business['categories']:
	# 					business_info.append({'business_id': business['business_id'], 'rating': business['stars']})
	# 					ids.append(business['business_id'])
	# 					count += 1
	# 		if count >= business_count:
	# 			break

	# print('businesses found : %d'%len(ids))

	# review_info = []
	# count = 0
	# with open('../yelp_academic_dataset_review.json') as review_json:
	#     for line in review_json:
	#     	count += 1
	#         review = json.loads(line)
	#         if review['business_id'] in ids:
	# 			review_info.append({'business_id': review['business_id'],
	# 								'rating' : review['stars'],
	#                                 'review': review['text'].replace('?', '')
	#                                 .replace(',', '').replace('.', '')})
	#         if count >= reviews_count:
	#         	break

	# print('reviews found : %d'%len(review_info))

	# business_info = pd.DataFrame(business_info)

	# review_info = pd.DataFrame(review_info)
	#review_info = pd.DataFrame(review_info.groupby('business_id')['review'] \
	                           # .apply(lambda x: x.sum())).reset_index()

	# info = pd.merge(business_info, review_info, how='inner', on='business_id')

	# info.head()

	# if wrt_csv == 1:
	# 	review_info.to_csv('business_train.csv', encoding='utf-8')

	review_info = pd.read_csv('../reviews.csv', encoding='ISO-8859-1')

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

	if verbose == 1:
		print("file read done")

	vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
	dtm = vectorizer.fit_transform(features)

	if verbose == 1:
		print("tfidf done")

	X = dtm
	y = labels
	X_train, X_test, y_train, y_test = train_test_split(dtm, labels, test_size=0.20)

	if verbose == 1:
		print("data sets established")

	preds = []

	hl_len = len(dtm[1])-5
	print("hl neurons : %d"%hl_len)

	# mlp_nn = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(hl_len,hl_len-10,hl_len-15))
	# 49%

	# mlp_nn = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(hl_len,hl_len-15))
	# 50%

	# mlp_nn = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(hl_len,hl_len-5,hl_len-10))
	# 42%

	# mlp_nn = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(hl_len,hl_len-20))
	# 50.3%

	mlp_nn = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(hl_len/2))

	mlp_nn.fit(X_train, y_train)

	if verbose == 1:
		print("NN trained")

	preds = mlp_nn.predict(X_test)

	good = 0
	count = 0

	print('predict -- real')
	for i in range(len(preds)-10):
		count += 1
		# print(preds[i+1] + '  --  ' + y_test[i+1])
		if preds[i+1] == y_test[i+1]:
			# print('good')
			good += 1
		# else:
		# 	print('XXX')

	print('good  : %d'%good)
	print('total : %d'%count)


	# print(X[11])
	# print(y[11])

	# print(X[42])
	# print(y[42])

main()