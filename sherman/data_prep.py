import json
import pandas as pd

# Helper function to clean text data
def clean_text(s):
    html_esc = ['&lt;', '&gt;', '&amp;']
    sp_symbols = ['\t', '\n', '\r']
    for sym in html_esc + sp_symbols:
        s = s.replace(sym, ' ')
    symbols = '1234567890!"#$%&\'()*+,-./:;?@[]^_`{|}~'
    for sym in symbols:
        s = s.replace(sym, ' ')
    return s

# List of cities to consider
cities = ['Phoenix']

# Business dataset
ids = []
business_info = []
with open('yelp_academic_dataset_business.json', encoding='utf8') as business_json:
    for line in business_json:
        business = json.loads(line)
        if business['city'] in cities:
            if business['categories'] is not None:
                if 'Restaurants' in business['categories']:
                    ids.append(business['business_id'])
                    business_info.append({'business_id': business['business_id'], 'rating': business['stars']})
        
# Review dataset
review_info = []
with open('yelp_academic_dataset_review.json', encoding='utf8') as review_json:
    for line in review_json:
        review = json.loads(line)
        if review['business_id'] in ids:
            review_info.append({'business_id': review['business_id'],
                                'review': clean_text(review['text']),
                                'rating': review['stars']})

# Create dataframes
business_info = pd.DataFrame(business_info)
review_info = pd.DataFrame(review_info)

# Save to CSV
review_info.to_csv('reviews.csv', index=False)