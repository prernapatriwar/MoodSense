import pandas as pd

# Load the training, testing, and validation datasets
train_data = pd.read_csv(r'/Users/kaustubhmestri/Projects/MoodSense/data/test.txt', sep=';', names=['text','emotion'])
test_data = pd.read_csv(r'/Users/kaustubhmestri/Projects/MoodSense/data/train.txt', sep=';', names=['text','emotion'])
val_data = pd.read_csv(r'/Users/kaustubhmestri/Projects/MoodSense/data/val.txt', sep=';', names=['text','emotion'])

# Check the first few rows
print(train_data.head())

import nltk
import re

# Download stopwords for text cleaning
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to clean the text
def clean_text(text):
    text = text.lower()  # lowercase text
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-alphabetic characters
    return text

# Apply cleaning function to the text data
train_data['cleaned_text'] = train_data['text'].apply(clean_text)
test_data['cleaned_text'] = test_data['text'].apply(clean_text)
val_data['cleaned_text'] = val_data['text'].apply(clean_text)

# Check cleaned data
print(train_data.head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(train_data['cleaned_text'])

# Transform the test and validation data
X_test_tfidf = tfidf.transform(test_data['cleaned_text'])
X_val_tfidf = tfidf.transform(val_data['cleaned_text'])

# Check the shape of the transformed data
print(X_train_tfidf.shape)
print(X_test_tfidf.shape)
print(X_val_tfidf.shape)

from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder on the training labels and transform them
y_train = label_encoder.fit_transform(train_data['emotion'])
y_test = label_encoder.transform(test_data['emotion'])
y_val = label_encoder.transform(val_data['emotion'])

# Check label encoding
print(label_encoder.classes_)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, 'model/sentiment_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'model/tfidf.pkl')

# Save label encoder (optional but useful for inference)
joblib.dump(label_encoder, 'model/label_encoder.pkl')
