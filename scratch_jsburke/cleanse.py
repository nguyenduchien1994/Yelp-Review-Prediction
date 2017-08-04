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

info.to_csv('yelp_extract.csv')
# print("business id :  raing")
# for business in business_info:
#	 print("%s : %.2f" % (business['business_id'], business['rating']))



		# thoughts meanderings, super tired, to be with john

		# if 'Restaurants' in business['categories']:
		# 	# print('Restaurants' in business['categories'])
		# 	print(True)
		# else:
		# 	continue

		#if business['stars'] >= 3.6:
		#	last_good = business

# print(last_good)
# type(last_good['categories'])

		#print('Restaurants' in business['categories'])

		# if 'Restaurants' in business['categories']:
		# 	ids.append(business['business_id'])
		# 	business_info.append({'business_id': business['business_id'], 'rating': business['stars']})