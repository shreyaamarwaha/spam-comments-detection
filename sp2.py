import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('/Users/shreya/Downloads/archive-2/Youtube01-Psy.csv')

# Print the first few rows of the dataset
print(df.head())

# Count the total number of spam comments in the dataset
total_spam_comments = df[df['CLASS'] == 1].shape[0]
print(f'Total number of spam comments in the dataset: {total_spam_comments}')

# Data Preprocessing

# Applying preprocessing to each comment in the 'CONTENT' column
df['cleaned_text'] = df['CONTENT'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.lower().split())

# Initializing stopwords and apply them to each comment
stop_words = set(stopwords.words('english'))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Initializing Porter Stemmer and apply it to each comment
stemmer = PorterStemmer()
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: [stemmer.stem(word) for word in x])

# After stemming, joining the cleaned words back into sentences
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(x))

# Encoding labels using LabelEncoder
label_encoder = LabelEncoder()
df['CLASS'] = label_encoder.fit_transform(df['CLASS'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['CLASS'], test_size=0.2, random_state=42)

# Initializing the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Transforming the training and testing data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Building and Training the Model

# Initializing and training the Multinomial Naive Bayes classifier
spam_classifier = MultinomialNB()
spam_classifier.fit(X_train_tfidf, y_train)

# Evaluating the Model

# Predicting labels for the test data
y_pred = spam_classifier.predict(X_test_tfidf)

# Calculating accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion)
print('Classification Report:')
print(report)

# Count the number of spam comments detected in the test set
detected_spam_comments = sum(y_pred)
print(f'Number of spam comments detected in the test set: {detected_spam_comments}')
