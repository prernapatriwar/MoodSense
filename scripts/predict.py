import joblib
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download stopwords for text processing
nltk.download('stopwords')

# Load the trained model, TF-IDF vectorizer, and label encoder
model = joblib.load('/Users/kaustubhmestri/Projects/MoodSense/model/sentiment_model.pkl')
tfidf = joblib.load('/Users/kaustubhmestri/Projects/MoodSense/model/label_encoder.pkl')
label_encoder = joblib.load('/Users/kaustubhmestri/Projects/MoodSense/model/tfidf.pkl')

# Function to clean incoming text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

# Predict function
def predict(text):
    processed_text = preprocess_text(text)  # Preprocess the input text
    tfidf_input = tfidf.transform([processed_text])  # Transform text into TF-IDF

    # Make the prediction
    prediction = model.predict(tfidf_input)
    prediction_prob = model.predict_proba(tfidf_input)

    # Decode the predicted label back to emotion
    predicted_emotion = label_encoder.inverse_transform(prediction)

    result = {
        'emotion': predicted_emotion[0],
        'probability': prediction_prob[0].tolist()  # Probabilities of all classes
    }

    return result

# Example usage
if __name__ == '__main__':
    text = input("Enter the text to predict emotion: ")  # Get user input
    result = predict(text)
    print(f"Predicted Emotion: {result['emotion']}")
    print(f"Prediction Probability: {result['probability']}")
