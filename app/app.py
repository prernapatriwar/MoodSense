from flask import Flask, request, jsonify
import joblib
import nltk
import re
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Download stopwords for text processing
nltk.download('stopwords')

# Load the trained model, TF-IDF vectorizer, and label encoder
model = joblib.load('model/sentiment_model.pkl')
tfidf = joblib.load('model/tfidf.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

# Function to clean incoming text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove non-alphabetic characters
    return text

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text input from the POST request
        input_text = request.json['text']
        
        # Preprocess the input text
        processed_text = preprocess_text(input_text)
        tfidf_input = tfidf.transform([processed_text])  # Convert the text to TF-IDF
        
        # Get the prediction from the model
        prediction = model.predict(tfidf_input)
        prediction_prob = model.predict_proba(tfidf_input)  # Get the probabilities for all labels

        # Decode the predicted label back to emotion
        predicted_emotion = label_encoder.inverse_transform(prediction)

        # Return the result as a JSON response
        result = {
            'emotion': predicted_emotion[0],  # Predicted emotion
            'probability': prediction_prob[0].tolist()  # List of probabilities for all classes
        }
        return jsonify(result)

# Home route
@app.route('/')
def home():
    return "Welcome to the Mood Prediction API!"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
