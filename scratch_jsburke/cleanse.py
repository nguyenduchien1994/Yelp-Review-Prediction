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
rest_ctr = 0

with open("../yelp_academic_dataset_business.json") as business_json:
	for line in business_json:
		business = json.loads(line)
		if business['categories'] is not None:
			if 'Restaurants' in business['categories']:
				bus_app(business)
			elif 'Food' in business['categories']:
				bus_app(business)

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