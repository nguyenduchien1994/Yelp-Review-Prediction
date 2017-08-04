#!/usr/bin/env python

import json
import numpy as numpy
import scipy
import pandas as pd 
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer

ids = []
business_info = []

with open("../yelp_academic_dataset_business.json") as business_json:
	for line in business_json:
		business = json.loads(line)
		if business['categories'] is not None:
			if 'Restaurants' in business['categories']:
				business_info.append({'business_id': business['business_id'], 'rating': business['stars']})
				ids.append(business['business_id'])
			elif 'Food' in business['categories']:
				business_info.append({'business_id': business['business_id'], 'rating': business['stars']})
				ids.append(business['business_id'])

review_info = []
with open('../yelp_academic_dataset_review.json') as review_json:
    for line in review_json:
        review = json.loads(line)
        if review['business_id'] in ids:
            review_info.append({'business_id': review['business_id'],
                                'review': review['text'].replace('?', '')
                                .replace(',', '').replace('.', '')})

business_info = pd.DataFrame(business_info)

review_info = pd.DataFrame(review_info)
review_info = pd.DataFrame(review_info.groupby('business_id')['review'] \
                           .apply(lambda x: x.sum())).reset_index()

info = pd.merge(business_info, review_info, how='inner', on='business_id')

info.head()

vectorizer = TfidfVectorizer(stop_words='english', min_df=0.1, max_df=0.8,
                             max_features=1000, ngram_range=(1,2))
dtm = vectorizer.fit_transform(info.review.values).toarray()

X = dtm
le = preprocessing.LabelEncoder()
y = le.fit_transform(info.rating.values)
train_set_size = 19979
X_train = X[:train_set_size]
y_train = y[:train_set_size]
X_test = X[train_set_size:]
y_test = y[train_set_size:]
