import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Read from CSV file instead of processing JSON again
review_info = pd.read_csv('reviews.csv', encoding='ISO-8859-1')

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
features, labels = shuffle(features, labels, random_state=0)

# Convert reviews to a matrix of TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
dtm = vectorizer.fit_transform(features)

# Use ~80% of data for training and ~20% for testing
X = dtm
y = labels
X_train, X_test, y_train, y_test = train_test_split(dtm, labels, test_size=0.20)

# SVM
classifier = LinearSVC(C=1)
classifier.fit(X_train, y_train)
y_pred_test = classifier.predict(X_test)

# Accuracy on test set
print("Accuracy:", classifier.score(X_test, y_test))
