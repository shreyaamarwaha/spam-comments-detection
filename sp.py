import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/Users/shreya/Downloads/archive-2/Youtube01-Psy.csv')

print(df.head())

# Step 3: Data Preprocessing 

# Applying preprocessing to each comment in the 'comment' column
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
df['COMMENT_ID'] = label_encoder.fit_transform(df['COMMENT_ID'])

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['COMMENT_ID'], test_size=0.2, random_state=42)

# Initializing the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  

# Transforming the training and testing data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

spam_classifier = MultinomialNB()
spam_classifier.fit(X_train_tfidf, y_train)


y_pred = spam_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion)
print('Classification Report:')
print(report)
